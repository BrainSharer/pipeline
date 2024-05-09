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
from rechunker import rechunk
import dask
import dask.array as da
import numpy as np
from timeit import default_timer as timer
from distributed import Client, progress
from dask.diagnostics import ProgressBar
from library.image_manipulation.image_manager import ImageManager
from library.utilities.dask_utilities import aligned_coarse_chunks, get_optimum_chunks, get_store, get_store_from_path, get_transformations, imreads, mean_dtype
from library.utilities.utilities_process import SCALING_FACTOR, get_cpus, get_scratch_dir

class OmeZarrManager():
    """
    Manages the creation of OmeZarr files by performing necessary setup, generating transformations, and writing the data to the file.
    Chunk size is very important. On the first mip, set it so the store has 1,2048,2048 chunks.
    This works best. The chunks in later mips then get set to a lower size for better viewing in neuroglancer.

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
            Creating new store from previous shape=(479, 984, 1760) previous chunks=(8, 512, 512)
            New store at mip=0 with shape=(479, 492, 880) resized chunks=(8, 256, 256) and storing chunks=[64, 64, 64]

            """
        tmp_dir = get_scratch_dir()
        self.tmp_dir = os.path.join(tmp_dir, f'{self.animal}')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.xy_resolution = self.sqlController.scan_run.resolution
        self.z_resolution = self.sqlController.scan_run.zresolution
        if self.downsample:
            # Main mip took 10.05 seconds with chunks=[1, 256, 256] trimto=8
            # Main mip took 4.97 seconds with chunks=[1, 256, 256] trimto=8
            # Main mip took 4.87 seconds with chunks=[1, 256, 256] trimto=1
            self.trimto = 1
            self.storefile = 'C1T.zarr'
            self.rechunkmefile = 'C1T_rechunk.zarr'
            self.scaling_factor = SCALING_FACTOR
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
            self.chunks = [
                    [64, 64, 64],
                    [32, 32, 32],
                    [32, 32, 32],
                    [32, 32, 32],
                ]
            self.mips = len(self.chunks)
            #self.mips = 0
        else:
            # 
            self.trimto = 64
            self.storefile = 'C1.zarr'
            self.rechunkmefile = 'C1_rechunk.zarr'
            self.scaling_factor = 1
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
            self.chunks = [
                    [64, 128, 128],
                    [64, 64, 64],
                    [64, 64, 64],
                    [32, 32, 32],
                    [32, 32, 32],
                    [32, 32, 32],
                ]
            self.mips = len(self.chunks)
            self.mips = 0

        image_manager = ImageManager(self.input)
        self.ndims = image_manager.ndim
        self.dtype = image_manager.dtype

        self.storepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", self.storefile
        )
        self.rechunkmepath = os.path.join(
            self.tmp_dir, self.rechunkmefile
        )
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
            },
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
        transformations = get_transformations(self.axes, self.mips + 1)
        for transformation in transformations:
            print(transformation)

        jobs = 1
        GB = (psutil.virtual_memory().free // 1024**3) * 0.8
        workers, _ = get_cpus()

        if self.debug:
            if not self.downsample:
                print('Full resolution images must be run under dask distributed')
                sys.exit()
            self.write_first_resolution(client=None)
            self.rechunkme(client=None)
            for mip in range(0, self.mips):
                self.write_mips(mip, client=None)
            self.cleanup()
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

                    with Client(n_workers=workers, threads_per_worker=jobs) as client:
                        self.write_first_resolution(client)
                        self.rechunkme(client=client)
                        for mip in range(0, self.mips):
                            self.write_mips(mip, client)

            except Exception as ex:
                print('Exception in running builder in omezarr_manager')
                print(ex)


        self.build_zattrs(transformations)
        self.cleanup()

    def write_first_resolution(self, client=None):
        """
        Writes the first MIP (Maximum Intensity Projection) of the input image stack to a Zarr store.

        Args:
            client (dask.distributed.Client, optional): Dask distributed client for distributed computing. Defaults to None.

        Returns:
            None
        """
        start_time = timer()
        store = get_store(self.rechunkmepath, 0)
        if os.path.exists(os.path.join(self.storepath, 'scale0')):
            print('Rechunked store exists, no need to write full resolution.\n')
            return
        
        print('Building stack and zarr data.')

        tiff_stack = imreads(self.input)
        """
        
        if self.ndims == 3:
            tiff_stack = tiff_stack[:, 0:new_shape[1], 0:new_shape[2], ...]
            optimum_chunks = [1, tiff_stack.shape[1], tiff_stack.shape[2], 3]
            tiff_stack.rechunk(optimum_chunks)
            tiff_stack = np.moveaxis(tiff_stack, -1, 0)
           
            #####tiff_stack = np.expand_dims(tiff_stack, axis=0)
        #optimum_chunks = [1, tiff_stack.shape[1], tiff_stack.shape[2]]  
        """

        old_shape = tiff_stack.shape
        new_shape = aligned_coarse_chunks(old_shape[:3], 64)
        tiff_stack = tiff_stack[:, 0:new_shape[1], 0:new_shape[2]]
        tiff_stack = tiff_stack.rechunk('auto')
        print(f'tiff_stack shape={tiff_stack.shape} tiff_stack.chunksize={tiff_stack.chunksize}')
        z = zarr.zeros(tiff_stack.shape, chunks=[1, 1024, 1024], store=store, overwrite=True, dtype=self.dtype)
        if client is None:
            to_store = da.store(tiff_stack, z, lock=True, compute=True)
        else:
            to_store = da.store(tiff_stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Main mip took {total_elapsed_time} seconds with chunks={tiff_stack.chunksize}")

    def rechunkme(self, client=None):
        if os.path.exists(os.path.join(self.storepath, 'scale0')):
            print('Rechunked store exists, no need to rechunk full resolution.\n')
            return


        read_storepath = os.path.join(self.rechunkmepath, 'scale0')
        write_storepath = os.path.join(self.storepath, 'scale0')
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

        rechunkme_stack = da.from_zarr(url=read_storepath)
        print(f'Using rechunking store with shape={rechunkme_stack.shape} chunks={rechunkme_stack.chunksize}')
        leading_chunk = rechunkme_stack.shape[0]
        target_chunks = (64, 128, 128)
        if leading_chunk < target_chunks[0]:
            target_chunks = (leading_chunk, 128, 128)
        GB = (psutil.virtual_memory().free // 1024**3) * 0.8
        max_mem = f"{GB}GB"

        target_store = os.path.join(self.tmp_dir, "rechunked.zarr")
        temp_store = os.path.join(self.tmp_dir, "rechunked-tmp.zarr")
        array_plan = rechunk(
            rechunkme_stack, target_chunks, max_mem, target_store, temp_store=temp_store
        )
        with ProgressBar():
            rechunked = array_plan.execute()        
        rechunked = da.from_zarr(rechunked)
        store = get_store(self.storepath, 0)
        z = zarr.zeros(rechunked.shape, chunks=target_chunks, store=store, overwrite=True, dtype=self.dtype)
        if client is None:
            to_store = da.store(rechunked, z, lock=True, compute=True)
        else:
            to_store = da.store(rechunked, z, lock=False, compute=False)
            to_store = progress(client.compute(to_store))
            to_store = client.gather(to_store)
        print(f'Wrote rechunked data to: {write_storepath}')
        print()



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
        print(f'Using previous store with shape={previous_stack.shape} previous chunks={previous_stack.chunksize}')
        axis_dict = {0:self.axis_scales[0], 1:self.axis_scales[1], 2:self.axis_scales[2]}
        scaled_stack = da.coarsen(mean_dtype, previous_stack, axis_dict, trim_excess=True)
        print(f'Coarsened stack shape={scaled_stack.shape} coarsened chunks={scaled_stack.chunksize}')

        optimum_chunks = self.chunks[mip]
        scaled_stack = scaled_stack.rechunk(optimum_chunks)
        print(f'Creating new store at mip={mip} with shape={scaled_stack.shape} resized chunks={scaled_stack.chunksize} and storing chunks={optimum_chunks}')

        store = get_store(self.storepath, mip+1)
        z = zarr.zeros(scaled_stack.shape, chunks=optimum_chunks, store=store, overwrite=True, dtype=scaled_stack.dtype)
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
            z, y, x = transformation['scale']

            scale["coordinateTransformations"] = [{
                "scale": [z, y, x],
                "type": "scale",
            }]

            scale["path"] = f'scale{mip}'

            datasets.append(scale)

        multiscales["datasets"] = datasets

        multiscales["type"] = (1, 2, 2)

        # Define down sampling methods for inclusion in zattrs
        description = '(1,1,2,2) downsample of in up to 3 dimensions calculated using the local mean'
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
