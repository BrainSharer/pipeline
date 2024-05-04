"""
Place the yaml below in: ~/.config/dask/distributed.yaml

distributed:
  worker:
    # Fractions of worker memory at which we take action to avoid memory blowup
    # Set any of the lower three values to False to turn off the behavior entirely
    memory:
      target: 0.50  # target fraction to stay below
      spill: 0.60  # fraction at which we spill to disk
      pause: 0.70  # fraction at which we pause worker threads
      terminate: False  # fraction at which we terminate the worker

"""
import glob
import os
import sys
import shutil
import psutil
import zarr
import dask
import dask.array as da
import numpy as np
from timeit import default_timer as timer
from distributed import Client, progress
from library.image_manipulation.image_manager import ImageManager
from library.utilities.dask_utilities import aligned_coarse_chunks, get_store, get_store_from_path, get_transformations, imreads, mean_dtype
from library.utilities.utilities_process import SCALING_FACTOR, get_cpus, get_scratch_dir

class OmeZarrManager():
    """
    Manages the creation of OmeZarr files by performing necessary setup, generating transformations, and writing the data to the file.

    Attributes:
        tmp_dir (str): The temporary directory for storing intermediate files.
        xy_resolution (float): The resolution of the xy plane.
        z_resolution (float): The resolution of the z axis.
        trimto (int): The trimto value based on the downsample flag.
        storefile (str): The name of the store file.
        scaling_factor (int): The scaling factor based on the downsample flag.
        input (str): The input file location.
        chunks (list): The list of chunk sizes for each MIP level.
        initial_chunks (list): The initial chunk size.
        mips (int): The number of MIP levels.
        ndims (int): The number of dimensions of the input image.
        storepath (str): The path to the store file.
        axes (list): The list of axis configurations.
        axis_scales (list): The list of coarsen values for each axis.
    """

    def setup(self):
            """
            Set up the omezarr manager by initializing necessary variables and configurations.

            This method performs the following steps:
            1. Creates a temporary directory for storing intermediate files.
            2. Sets the xy_resolution and z_resolution based on the scan run.
            3. Sets the trimto, storefile, scaling_factor, input, chunks, initial_chunks, and mips based on the downsample flag.
            4. Determines the number of dimensions of the input image.
            5. Sets the storepath and axes based on the file locations and resolutions.
            6. Sets the axis_scales based on the coarsen values of the axes.

            """
            tmp_dir = get_scratch_dir()
            self.tmp_dir = os.path.join(tmp_dir, f'{self.animal}')
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
            self.xy_resolution = self.sqlController.scan_run.resolution
            self.z_resolution = self.sqlController.scan_run.zresolution
            if self.downsample:
                self.trimto = 8
                self.storefile = 'C1T.zarr'
                self.scaling_factor = SCALING_FACTOR
                self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
                self.chunks = [
                    [1, 1024, 1024],
                    [16, 128, 128],
                    [32, 64, 64],
                    [64, 64, 64],
                ]
                self.initial_chunks = [1, 1024, 1024]
                self.mips = len(self.chunks)
            else:
                self.trimto = 64
                self.storefile = 'C1.zarr'
                self.scaling_factor = 1
                self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
                self.chunks = [
                    [4, 1024, 1024],
                    [8, 512, 512],
                    [16, 256, 256],
                    [32, 128, 128],
                    [64, 64, 64],
                    [64, 64, 64],
                    [64, 64, 64],
                    [32, 32, 32]
                ]
                self.initial_chunks = [1, 2048, 2048]
                self.mips = len(self.chunks)

            image_manager = ImageManager(self.input)
            self.ndims = image_manager.ndim
            if self.ndims == 3:
                self.initial_chunks.append(3)

            self.storepath = os.path.join(self.fileLocationManager.www, 'neuroglancer_data', self.storefile)
            self.axes = [
                {
                    "name": "z",
                    "type": "space",
                    "unit": "micrometer",
                    "coarsen": 1,
                    "resolution": self.z_resolution,
                },
                {
                    "name": "y",
                    "type": "space",
                    "unit": "micrometer",
                    "coarsen": 2,
                    "resolution": self.xy_resolution * self.scaling_factor,
                },
                {
                    "name": "x",
                    "type": "space",
                    "unit": "micrometer",
                    "coarsen": 2,
                    "resolution": self.xy_resolution * self.scaling_factor,
                }
            ]
            self.axis_scales = [a["coarsen"] for a in self.axes]

    def create_omezarr(self):
            """
            Creates an omezarr file by performing the necessary setup, generating transformations,
            and writing the data to the file.

            Returns:
                None
            """
            self.setup()
            transformations = get_transformations(self.axes, self.mips)
            for transformation in transformations:
                print(transformation)

            jobs = 1
            GB = (psutil.virtual_memory().free // 1024**3) * 0.8
            workers, _ = get_cpus()

            if self.debug:
                self.write_first_mip(client=None)
                for mip in range(0, self.mips):
                    self.write_mips(mip, client=None)
            else:

                try:
                    with dask.config.set({'temporary_directory': self.tmp_dir, 
                                        'logging.distributed': 'error',
                                        'distributed.comm.retry.count': 10,
                                        'distributed.comm.timeouts.connect': 30}):

                        print(f'Starting distributed dask with {workers} workers and {jobs} jobs in tmp dir={self.tmp_dir} with free memory={GB}GB')
                        print('With Dask memory config:')
                        print(dask.config.get("distributed.worker.memory"))
                        print()

                        # https://github.com/dask/distributed/blob/main/distributed/distributed.yaml#L129-L131
                        os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
                        os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
                        os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "60s"
                        # https://docs.dask.org/en/stable/array-best-practices.html#orient-your-chunks
                        os.environ["OMP_NUM_THREADS"] = "1"
                        os.environ["MKL_NUM_THREADS"] = "1"
                        os.environ["OPENBLAS_NUM_THREADS"] = "1"

                        with Client(n_workers=workers, threads_per_worker=jobs) as client:
                            self.write_first_mip(client)
                            for mip in range(0, self.mips):
                                    self.write_mips(mip, client)

                except Exception as ex:
                    print('Exception in running builder in omezarr_manager')
                    print(ex)

            self.build_zattrs(transformations)
            self.cleanup()

    def write_first_mip(self, client=None):
        """
        Writes the first MIP (Maximum Intensity Projection) of the input image stack to a Zarr store.

        Args:
            client (dask.distributed.Client, optional): Dask distributed client for distributed computing. Defaults to None.

        Returns:
            None
        """
        start_time = timer()
        store = get_store(self.storepath, 0)
        if os.path.exists(os.path.join(self.storepath, 'scale0')):
            print('Initial store exists, continuing\n')            
            return
        print('Building Virtual Stack')

        tiff_stack = imreads(self.input)
        old_shape = tiff_stack.shape
        new_shape = aligned_coarse_chunks(old_shape, self.trimto)
        tiff_stack = tiff_stack[:, 0:new_shape[1], 0:new_shape[2]]
        optimum_chunks = [1, tiff_stack.shape[1], tiff_stack.shape[2]]

        tiff_stack.rechunk(optimum_chunks)
        print(f'tiff_stack shape={tiff_stack.shape} tiff_stack.chunksize={tiff_stack.chunksize} stored chunks={self.initial_chunks}')

        z = zarr.zeros(tiff_stack.shape, chunks=self.initial_chunks, store=store, overwrite=True, dtype=np.uint16)
        if client is None:
            to_store = da.store(tiff_stack, z, lock=True, compute=True)
        else:
            to_store = da.store(tiff_stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Main mip took {total_elapsed_time} seconds with chunks={self.initial_chunks} trimto={self.trimto}")


    def write_mips(self, mip, client=None):
            """
            Writes multi-resolution image pyramids (MIPs) to the specified storepath.

            Args:
                mip (int): The MIP level to write.
                client (dask.distributed.Client, optional): The Dask distributed client to use for computation. Defaults to None.

            Raises:
                SystemExit: If the read_storepath does not exist.

            Returns:
                None
            """
            read_storepath = os.path.join(self.storepath, f'scale{mip}')
            write_storepath = os.path.join(self.storepath, f'scale{mip+1}')
            print(f'Loading data at: {read_storepath}', end=" ")
            if os.path.exists(read_storepath):
                print(': Success!')
            else:
                print('\nError: exiting ...')
                print(f'Missing {read_storepath}')            
                sys.exit()

            if os.path.exists(write_storepath):
                print(f'Already exists: {write_storepath}')            
                return

            previous_stack = da.from_zarr(url=read_storepath)
            print(f'Creating new store from previous shape={previous_stack.shape} previous chunks={previous_stack.chunksize}')
            axis_dict = {0:self.axis_scales[0], 1:self.axis_scales[1], 2:self.axis_scales[2]}
            scaled_stack = da.coarsen(mean_dtype, previous_stack, axis_dict, trim_excess=True)

            chunks = self.chunks[mip]
            # optimum_chunks = optimize_chunk_shape(scaled_stack.shape, scaled_stack.chunksize, chunks)
            optimum_chunks = [1, scaled_stack.shape[1], scaled_stack.shape[2]]
            scaled_stack.rechunk(optimum_chunks)
            print(f'New store at mip={mip} with shape={scaled_stack.shape} resized chunks={scaled_stack.chunksize} and storing chunks={chunks}')

            store = get_store(self.storepath, mip+1)
            z = zarr.zeros(scaled_stack.shape, chunks=chunks, store=store, overwrite=True, dtype=scaled_stack.dtype)
            if client is None:
                da.store(scaled_stack, z, lock=True, compute=True)
            else:
                to_store = da.store(scaled_stack, z, lock=False, compute=False)
                to_store = progress(client.compute(to_store))
                to_store = client.gather(to_store)
            print(f'Wrote mip with data to: {write_storepath}')
            print()

    def build_zattrs(self, transformations):
        """
        Build the zarr attributes for the given transformations.

        Args:
            transformations (list): A list of dictionaries representing the transformations.

        Returns:
            None
        """
        store = get_store_from_path(self.storepath)
        r = zarr.open(store)        
        multiscales = {}
        multiscales["version"] = "0.5-dev"
        multiscales["name"] = self.animal

        multiscales["axes"] = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ]

        datasets = [] 
        for mip, transformation in enumerate(transformations):
            scale = {}

            scale["coordinateTransformations"] = [{
                "scale": transformation['scale'],
                "type": "scale",
            }]

            scale["path"] = f'scale{mip}'

            datasets.append(scale)

        multiscales["datasets"] = datasets

        multiscales["type"] = (1, 2, 2)

        # Define down sampling methods for inclusion in zattrs
        description = '(1,2,2) downsample of in up to 3 dimensions calculated using the local mean'
        details = 'downsampling is done using dask coarsen and numpy mean'

        multiscales["metadata"] = {
                        "description": description,
                        "details": details
                    }

        r.attrs['multiscales'] = [multiscales]

        omero = {}
        omero['id'] = 1
        omero['name'] = self.animal
        omero['version'] = "0.5-dev"
        colors = None
        # colors = self.omero_dict['channels']['color']
        # If the colors dict is not present, then make it with a rotating color palette
        if colors is None:
            colors_palette = [
                'FF0000', #red
                '00FF00', #green
                'FF00FF', #purple
                'FF00FF'  #blue
                ]
            colors = {}
            for idx in range(1):
                colors[idx] = colors_palette[idx%len(colors_palette)]

        channels = []
        for chn in range(1):
            new = {}
            new["active"] = True
            new["coefficient"] = 1
            new["color"] = colors[chn]
            new["family"] = "linear"
            new['inverted'] = False
            new['label'] = self.channel

            end = mx = 2**16 - 1

            new['window'] = {
                "end": end,
                "max": mx,
                "min": 0,
                "start": 0
                }

            channels.append(new)

        omero['channels'] = channels

        omero['rdefs'] = {
            "defaultZ": 1,
            "model": "greyscale"                  # "color" or "greyscale"
            }

        r.attrs['omero'] = omero

    def cleanup(self):
            """
            Cleans up the temporary directory and removes orphaned lock files.

            This method removes all files and directories in the temporary directory
            and removes any .lock files in the output directory.

            Raises:
                KeyboardInterrupt: If the cleanup process is interrupted by the user.
                Exception: If an error occurs during the cleanup process.
            """
            countKeyboardInterrupt = 0
            countException = 0
            print('Cleaning up tmp dir and orphaned lock files')
            while True:
                try:
                    # Remove any existing files in the temp_dir
                    files = glob.glob(os.path.join(self.tmp_dir, "**/*"), recursive=True)
                    for file in files:
                        print(f'Removing {file}')
                        try:
                            if os.path.isfile(file):
                                os.remove(file)
                            elif os.path.isdir(file):
                                shutil.rmtree(file)
                        except Exception:
                            pass

                    # Remove any .lock files in the output directory (recursive)

                    print("Removing locks")
                    locks = glob.glob(os.path.join(self.storepath, "**/*.lock"), recursive=True)
                    for lock in locks:
                        try:
                            if os.path.isfile(lock):
                                print(f'Removing {lock}')
                                os.remove(lock)
                            elif os.path.isdir(lock):
                                shutil.rmtree(lock)
                        except Exception as ex:
                            print('Trouble removing lock')
                            print(ex)
                            pass

                    break
                except KeyboardInterrupt:
                    countKeyboardInterrupt += 1
                    if countKeyboardInterrupt == 4:
                        break
                    pass
                except Exception:
                    countException += 1
                    if countException == 100:
                        break
                    pass

            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
