import os
import glob, time, shutil
import psutil
import zarr
import dask

from library.omezarr.omezarr_init import OmeZarrBuilder
from library.utilities.dask_utilities import write_first_mip, write_mip_series
from library.utilities.utilities_process import SCALING_FACTOR

class OmeZarrManager():
    """"""

    def __init__(self):
        """
        """

    def create_omezarr(self):
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        if self.downsample:
            storefile = 'C1T.zarr'
            scaling_factor = SCALING_FACTOR
            INPUT = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        else:
            storefile = 'C1.zarr'
            scaling_factor = 1
            INPUT = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
        if not os.path.exists(INPUT):
            print(f'Missing: {INPUT}')
            return
        files = os.listdir(INPUT)
        if len(files) < 5:
            print(f'Not enough files in: {INPUT}')
            return
        # Open the zarr group manually
        storepath = os.path.join(self.fileLocationManager.www, 'neuroglancer_data', storefile)

        axes = [
            {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 2,
                "resolution": xy_resolution * scaling_factor,
            },
            {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 2,
                "resolution": xy_resolution * scaling_factor,
            },
            {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 1,
                "resolution": z_resolution,
            }
        ]
        axis_scales = [a["coarsen"] for a in axes]
        write_first_mip(INPUT, storepath)
        return
        
        try:
            with dask.config.set():  #<<-Disable WARNING messages that are often not helpful (remove for debugging)

                workers = 8
                threads = 1

                #https://github.com/dask/distributed/blob/main/distributed/distributed.yaml#L129-L131
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
                os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "60s"
                write_mip_series(INPUT, storepath)
                print('Fini')

        except Exception as ex:
            print('Exception in running builder in omezarr_manager')
            print(ex)
        
        

    def create_omezarrXXX(self):
        print('Ome zarr manager setup')
        INPUT = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        OUTPUT = os.path.join(self.fileLocationManager.www, 'neuroglancer_data', 'C1T.ome.zarr')
        scales = (1, 1, 20.0, 10.4, 10.4)
        originalChunkSize = (1, 1, 1, 64, 64)
        finalChunkSize = (1, 1, 64, 64, 64)
        #TODOscales = (20.0, 10.4, 10.4)
        #TODOoriginalChunkSize = (64, 64, 64)
        #TODOfinalChunkSize = (64, 64, 64)
        cpu_cores = os.cpu_count()
        mem=int((psutil.virtual_memory().free/1024**3)*.8)
        zarr_store_type=zarr.storage.NestedDirectoryStore
        tmp_dir='/tmp'
        debug=self.debug
        omero = {}
        omero['channels'] = {}
        omero['channels']['color'] = None
        omero['channels']['label'] = None
        omero['channels']['window'] = None
        omero['name'] = self.animal
        omero['rdefs'] = {}
        omero['rdefs']['defaultZ'] = None

        downSampleType='mean'

        omezarr_builder = OmeZarrBuilder(
            INPUT,
            OUTPUT,
            scales=scales,
            originalChunkSize=originalChunkSize,
            finalChunkSize=finalChunkSize,
            cpu_cores=cpu_cores,
            mem=mem,
            tmp_dir=tmp_dir,
            debug=debug,
            zarr_store_type=zarr_store_type,
            omero_dict=omero,
            downSampType=downSampleType,
        )

        try:
            #with dask.config.set({'temporary_directory': omezarr_builder.tmp_dir, #<<-Chance dask working directory
            #                      'logging.distributed': 'error'}):  #<<-Disable WARNING messages that are often not helpful (remove for debugging)
            with dask.config.set({'temporary_directory': omezarr_builder.tmp_dir}):  #<<-Disable WARNING messages that are often not helpful (remove for debugging)

                workers = omezarr_builder.workers
                threads = omezarr_builder.sim_jobs

                #https://github.com/dask/distributed/blob/main/distributed/distributed.yaml#L129-L131
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
                os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "60s"
                print('Building OME Zarr with workers {}, threads {}, mem {}, chunk_size_limit {}'.format(workers, threads, omezarr_builder.mem, omezarr_builder.res0_chunk_limit_GB))
                omezarr_builder.write_resolution_series()

        except Exception as ex:
            print('Exception in running builder in omezarr_manager')
            print(ex)


        #Cleanup
        countKeyboardInterrupt = 0
        countException = 0
        print('Cleaning up tmp dir and orphaned lock files')
        while True:
            try:
                #Remove any existing files in the temp_dir
                filelist = glob.glob(os.path.join(omezarr_builder.tmp_dir, "**/*"), recursive=True)
                for f in filelist:
                    try:
                        if os.path.isfile(f):
                            os.remove(f)
                        elif os.path.isdir(f):
                            shutil.rmtree(f)
                    except Exception:
                        pass

                #Remove any .lock files in the output directory (recursive)
                lockList = glob.glob(os.path.join(omezarr_builder.out_location, "**/*.lock"), recursive=True)
                for f in lockList:
                    try:
                        if os.path.isfile(f):
                            os.remove(f)
                        elif os.path.isdir(f):
                            shutil.rmtree(f)
                    except Exception:
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
