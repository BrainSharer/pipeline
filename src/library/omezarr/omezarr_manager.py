import os
import glob, time, shutil

from numcodecs import Blosc
import psutil
import zarr
import dask
from zarr_stores.archived_nested_store import Archived_Nested_Store
from zarr_stores.h5_nested_store import H5_Nested_Store

from library.omezarr.omezarr_init import OmeZarrBuilder

class OmeZarrManager():
    """"""

    def __init__(self):
        """
        """

    def create_omezarr(self):
        print('Ome zarr manager setup')
        INPUT = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        OUTPUT = os.path.join(self.fileLocationManager.www, 'neuroglancer_data', 'C1T.ome.zarr')
        #####scales = (1, 1, 20.0, 10.4, 10.4)
        #####originalChunkSize = (1, 1, 1, 64, 64)
        #####finalChunkSize = (1, 1, 32, 64, 64)
        scales = (20.0, 10.4, 10.4)
        originalChunkSize = (1, 64, 64)
        finalChunkSize = (64, 64, 64)
        cpu_cores = os.cpu_count()
        mem=int((psutil.virtual_memory().free/1024**3)*.8)
        zarr_store_type=zarr.storage.NestedDirectoryStore
        writeDirect=True
        tmp_dir='/tmp'
        debug=self.debug
        verify_zarr_write=False
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
            geometry=scales,
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
            with dask.config.set({'temporary_directory': omezarr_builder.tmp_dir, #<<-Chance dask working directory
                                  'logging.distributed': 'error'}):  #<<-Disable WARNING messages that are often not helpful (remove for debugging)

                workers = omezarr_builder.workers
                threads = omezarr_builder.sim_jobs

                #https://github.com/dask/distributed/blob/main/distributed/distributed.yaml#L129-L131
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
                os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "60s"
                print('Buding OME Zarr with workers {}, threads {}, mem {}, chunk_size_limit {}'.format(workers, threads, omezarr_builder.mem, omezarr_builder.res0_chunk_limit_GB))
                omezarr_builder.write_resolution_series()

        except Exception as ex:
            print('Exception')
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
