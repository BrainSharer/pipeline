"""
Place the yaml below in: ~/.config/dask/distributed.yaml

distributed:
  worker:
    # Fractions of worker memory at which we take action to avoid memory blowup
    # Set any of the lower three values to False to turn off the behavior entirely
    memory:
      target: 0.85  # target fraction to stay below
      spill: 0.86  # fraction at which we spill to disk
      pause: 0.87  # fraction at which we pause worker threads
      terminate: 0.9  # fraction at which we terminate the worker
"""
import os
import inspect
import dask
import dask.config
from distributed import LocalCluster    
from dask.distributed import Client
from timeit import default_timer as timer
from tqdm import tqdm
import zarr

from library.image_manipulation.image_manager import ImageManager
from library.omezarr.builder_init import builder
from library.utilities.dask_utilities import closest_divisors_to_target
from library.utilities.utilities_process import SCALING_FACTOR, write_image


class OmeZarrManager():
    """
    chunk size timings on downsampled images
    256 = omezarr took 102.05 seconds
    512 = omezarr took 44.06 seconds
    1024 = omezarr took 33.65 seconds
    """

    def get_omezarr_info(self):
        if self.downsample:
            storefile = f'C{self.channel}T.zarr'
        else:
            storefile = f'C{self.channel}.zarr'
        storepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", storefile
        )
        if os.path.exists(storepath):
            scale_paths = sorted(os.listdir(storepath))
            scale_paths = [os.path.join(storepath, p) for p in scale_paths if os.path.isdir(os.path.join(storepath, p))]
            print(f'{len(scale_paths)} OME-Zarr stores exist at {storepath}')
            for scale_path in scale_paths:
                if os.path.exists(scale_path) and os.path.isdir(scale_path):
                    print(f'Resolution {os.path.basename(scale_path)} exists at {scale_path}')                
                    store = store = zarr.storage.NestedDirectoryStore(scale_path)
                    volume = zarr.open(store, 'r')
                    print(volume.info)
                    print(f'volume.shape={volume.shape}')
        else:
            print(f'OME-Zarr store {storepath} does not exist')

    def write_sections_from_volume(self):
        start_time = timer()
        mip = str(self.zarrlevel)
        if self.downsample:
            zarrpath = os.path.join(self.fileLocationManager.neuroglancer_data, f'C{self.channel}T.zarr', mip)
        else:
            zarrpath = os.path.join(self.fileLocationManager.neuroglancer_data, f'C{self.channel}.zarr', mip)

        outpath = os.path.join( self.fileLocationManager.prep, f'C{self.channel}', mip)
        os.makedirs(outpath, exist_ok=True)
        if os.path.exists(zarrpath):
            print(f'Using existing {zarrpath}')
        else:
            print(f'No zarr: {zarrpath}')
            return

        os.makedirs(outpath, exist_ok=True)
        store = store = zarr.storage.NestedDirectoryStore(zarrpath)
        volume = zarr.open(store, 'r')

        if self.debug:
            print(f'Volume type ={type(volume)}')
            print(f'Volume shape={volume.shape} ')
            print(f'Volume dtype={volume.dtype}')
            print('Exiting early')
            return
        else:

            for i in tqdm(range(volume.shape[-1]), disable=self.debug):
                outfile = os.path.join(outpath, f'{str(i).zfill(4)}.tif')
                if os.path.exists(outfile):
                    continue
                section = volume[..., i]
                if section.ndim > 2:
                    section = section.reshape(section.shape[-2], section.shape[-1])

                write_image(outfile, section)

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'writing {i+1} sections in C{self.channel} completed in {total_elapsed_time} seconds"')



    def create_omezarr(self):
        """Create OME-Zarr (NGFF) data store. WIP
        """
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        """Set some OS variables"""
        os.environ['OMP_NUM_THREADS']='1'
        os.environ['MKL_NUM_THREADS']='1'
        os.environ['OPENBLAS_NUM_THREADS']='1'        
        
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        iteration = self.get_alignment_status()
        if iteration is None:
            print('No alignment iterations found.  Please run the alignment steps first.')
            return
        input, _ = self.fileLocationManager.get_alignment_directories(channel=self.channel, downsample=self.downsample, iteration=iteration)
        print(f'Creating OME-Zarr from {input}')
        image_manager = ImageManager(input)

        if self.downsample:
            storefile = f'C{self.channel}T.zarr'
            scaling_factor = SCALING_FACTOR
            chunk_y = image_manager.height
            mips = 3
        else:
            storefile = f'C{self.channel}.zarr'
            scaling_factor = 1  
            chunk_y = closest_divisors_to_target(image_manager.height, image_manager.height // 2)
            mips = 8


        originalChunkSize = (1, chunk_y, image_manager.width)
        files = []
        for file in sorted(os.listdir(input)):
            filepath = os.path.join(input, file)
            files.append(filepath)
        
        if self.debug:
            print(f'INPUT FOLDER: {input}')
            print(f'INPUT FILES COUNT: {len(files)}')

        omero = {}
        omero['channels'] = {}
        omero['channels']['color'] = None
        omero['channels']['label'] = None
        omero['channels']['window'] = None
        omero['name'] = self.animal
        omero['rdefs'] = {}
        omero['rdefs']['defaultZ'] = len(files) // 2
        omero_dict = omero

        storepath = os.path.join(self.fileLocationManager.www, "neuroglancer_data", storefile)
        xy = xy_resolution * scaling_factor
        resolution = (z_resolution, xy, xy)

        omezarr = builder(
            input,
            storepath,
            files,
            resolution,
            originalChunkSize=originalChunkSize,
            scratch_space=self.scratch_space,
            debug=self.debug,
            omero_dict=omero_dict,
            mips=mips,
            downsample=self.downsample,
            channel=self.channel,
        )
        dask.config.set({'logging.distributed': 'info', 'temporary_directory': self.scratch_space})
        n_workers = os.cpu_count() // 6
        threads_per_worker = 4
        #cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=self.available_memory)
        memory_limit = str(int(self.available_memory / n_workers)) + 'GB'
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        print(f"Using Dask cluster for transfer with {n_workers} workers and {threads_per_worker} threads/per worker with {self.available_memory} GB available memory")
        if self.debug:
            exit(1)

        with Client(cluster) as client:
            print(f"Client dashboard: {client.dashboard_link}")
            omezarr.write_transfer(client)

        #cluster.close()
        ## The number of workers needs to be reduced for the remainder of the process
        #n_workers = n_workers // 2 if n_workers > 2 else 1
        #threads_per_worker = 4
        #cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=self.available_memory)
        #print(f"Using Dask cluster for remainder with {n_workers} workers and {threads_per_worker} threads/per worker with {self.available_memory} HN available memory")
        
        with Client(cluster) as client:
            """Transfer complete, now rechunk to final chunk size and create mips
            """
            # Transfer 0
            print('Starting transfer 0')
            input_path = omezarr.transfer_path
            output_path = omezarr.rechunkme_path
            omezarr.write_rechunk_transfer(client, input_path, output_path, level=-1)
            # Transfer 1
            print('Starting transfer 1')
            input_path = omezarr.rechunkme_path
            output_path = os.path.join(omezarr.output, str(0))
            omezarr.write_rechunk_transfer(client, input_path, output_path, level=0)
        
            pyramids = len(omezarr.pyramidMap)
            for mip in range(1, pyramids):
                print(f'Creating mip {mip} of {pyramids}')
                omezarr.write_mips(mip, client)

        cluster.close()
        omezarr.cleanup()

 