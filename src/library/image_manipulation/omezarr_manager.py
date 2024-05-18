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
import os
import dask

from dask.distributed import Client
from library.omezarr.builder_init import builder
from library.utilities.utilities_process import SCALING_FACTOR, get_scratch_dir

class OmeZarrManager():

    def create_omezarr(self):
        tmp_dir = get_scratch_dir()
        tmp_dir = os.path.join(tmp_dir, f'{self.animal}')
        os.makedirs(tmp_dir, exist_ok=True)
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        if self.downsample:
            storefile = 'C1T.zarr'
            scaling_factor = SCALING_FACTOR
            input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
            mips = 4
            originalChunkSize = [1, 1, 1, 512, 512]
            finalChunkSize=(1, 1, 32, 32, 32)
        else:
            storefile = 'C1.zarr'
            scaling_factor = 1
            input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
            mips = 8
            originalChunkSize = [1, 1, 1, 2048, 2048]
            finalChunkSize=(1, 1, 64, 64, 64)
        # vars from stack to multi
        filesList = []
        for file in sorted(os.listdir(input)):
            filepath = os.path.join(input, file)
            filesList.append(filepath)
        filesList = [filesList]
        omero = {}
        omero['channels'] = {}
        omero['channels']['color'] = None
        omero['channels']['label'] = None
        omero['channels']['window'] = None
        omero['name'] = self.animal
        omero['rdefs'] = {}
        omero['rdefs']['defaultZ'] = len(filesList) // 2
        omero_dict = omero


        storepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", storefile
        )
        xy = xy_resolution * scaling_factor
        geometry = (1, 1, z_resolution, xy, xy)



        omezarr = builder(
            input,
            storepath,
            filesList,
            geometry=geometry,
            originalChunkSize=originalChunkSize,
            finalChunkSize=finalChunkSize,
            tmp_dir=tmp_dir,
            debug=self.debug,
            omero_dict=omero_dict,
            mips=mips
        )

        if self.debug:
            print(f'Starting debug non-dask with {omezarr.workers} workers and {omezarr.sim_jobs} sim_jobs with free memory={omezarr.mem}GB')
            omezarr.write_resolution_0(client=None)
            omezarr.cleanup()
        else:


            try:
                with dask.config.set({'temporary_directory': tmp_dir, 
                                        'logging.distributed': 'error'}):

                    os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "160s"
                    os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "160s"
                    os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "160s"
                    # https://docs.dask.org/en/stable/array-best-practices.html#orient-your-chunks
                    os.environ["OMP_NUM_THREADS"] = "1"
                    os.environ["MKL_NUM_THREADS"] = "1"
                    os.environ["OPENBLAS_NUM_THREADS"] = "1"

                    print('With Dask memory config:')
                    print(dask.config.get("distributed.worker.memory"))
                    print()
                    print(f'Starting distributed dask with {omezarr.workers} workers and {omezarr.sim_jobs} sim_jobs with free memory={omezarr.mem}GB')
                    #cluster = LocalCluster(n_workers=omezarr.workers, threads_per_worker=omezarr.sim_jobs, processes=False)
                    with Client(n_workers=omezarr.workers, threads_per_worker=omezarr.sim_jobs) as client:
                        omezarr.write_resolution_0(client)
                        for mip in range(1, len(omezarr.pyramidMap)):
                            omezarr.write_resolutions(mip, client)

            except Exception as ex:
                print('Exception in running builder in omezarr_manager')
                print(ex)

            finally:
                omezarr.cleanup()

