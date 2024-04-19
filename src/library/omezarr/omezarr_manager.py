import glob
import os
import shutil
import psutil
import zarr
import dask
import dask.array as da
from distributed import Client, progress

from library.omezarr.omezarr_init import OmeZarrBuilder
from library.utilities.dask_utilities import aligned_coarse_chunks, get_store, get_store_from_path, get_transformations, imreads, mean_dtype
from library.utilities.utilities_process import SCALING_FACTOR, get_cpus

class OmeZarrManager():
    """"""

    def setup(self):
        """Set up variables
        """
        low, high = get_cpus()
        self.workers = 4
        self.jobs = 4
        self.xy_resolution = self.sqlController.scan_run.resolution
        self.z_resolution = self.sqlController.scan_run.zresolution
        if self.downsample:
            self.storefile = 'C1T.zarr'
            self.scaling_factor = SCALING_FACTOR
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
            self.mips = 4
        else:
            self.storefile = 'C1.zarr'
            self.scaling_factor = 1
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
            self.mips = 7

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
        self.setup()
        transformations = get_transformations(self.axes, self.mips)

        """
        self.write_first_mip(client=None)

        for scale, transformation in enumerate(transformations):
            self.write_mips(scale, transformation, client=None)
        """

        
        try:
            with dask.config.set({'distributed.scheduler.worker-ttl': None,
                                  'logging.distributed': 'error'}):

                print(f'Starting distributed dask with {self.workers} workers and {self.jobs} jobs')
                #https://github.com/dask/distributed/blob/main/distributed/distributed.yaml#L129-L131
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"
                os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "60s"
                self.write_mip_series(transformations)
                

        except Exception as ex:
            print('Exception in running builder in omezarr_manager')
            print(ex)

        self.build_zattrs(transformations)

    
    def write_mip_series(self, transformations):
        """
        Make downsampled versions of dataset based on number of transformations
        Requies that a dask.distribuited client be passed for parallel processing
        """

        with Client(n_workers=self.workers, threads_per_worker=self.jobs) as client:
            self.write_first_mip(client)

        for scale, _ in enumerate(transformations):
            with Client(n_workers=self.workers, threads_per_worker=self.jobs) as client:
                self.write_mips(scale, client)


    def write_first_mip(self, client):

        store = get_store(self.storepath, 0)
        if os.path.exists(os.path.join(self.storepath, f'scale0')):
            print('Initial store exists, continuing')            
            return
        print('Building Virtual Stack')
        tiff_stack = imreads(self.input)
        print(f'Original tiff_stack shape={tiff_stack.shape}')
        old_shape = tiff_stack.shape
        trimto = 8
        new_shape = aligned_coarse_chunks(old_shape, trimto)
        tiff_stack = tiff_stack[:, 0:new_shape[1], 0:new_shape[2]]
        print(f'Aligned tiff_stack shape={tiff_stack.shape}')
        tiff_stack = tiff_stack.rechunk('auto')
        #chunks = [128, 128, 128]
        chunks = tiff_stack.chunksize
        print(f'Setting up zarr store for main resolution with chunks={chunks}')
        z = zarr.zeros(tiff_stack.shape, chunks=chunks, store=store, overwrite=True, dtype=tiff_stack.dtype)

        to_store = da.store(tiff_stack, z, lock=False, compute=False)
        to_store = progress(client.compute(to_store))
        to_store = client.gather(to_store)
        print('Finished doing zarr store for main resolution\n')

    def write_mips(self, scale, client):
        read_storepath = os.path.join(self.storepath, f'scale{scale}')
        write_storepath = os.path.join(self.storepath, f'scale{scale + 1}')
        print(f'Checking if data exists at: {read_storepath}', end=" ")
        if os.path.exists(read_storepath):
            print(f'and loading data')
        else:
            print(f'and no store exists so returning ...')            
            return
        
        if os.path.exists(write_storepath):
            print(f'Already exists: {write_storepath}')            
            return

        previous_stack = da.from_zarr(url=read_storepath)
        print(f'Creating new store from previous shape={previous_stack.shape} chunks={previous_stack.chunksize}')
        axis_dict = {0:self.axis_scales[0], 1:self.axis_scales[1], 2:self.axis_scales[2]}
        scaled_stack = da.coarsen(mean_dtype, previous_stack, axis_dict, trim_excess=True)
        scaled_stack.rechunk('auto')
        chunks = scaled_stack.chunksize
        print(f'New store with shape={scaled_stack.shape} chunks={chunks}')

        store = get_store(self.storepath, scale + 1)
        z = zarr.zeros(scaled_stack.shape, chunks=chunks, store=store, overwrite=True, dtype=scaled_stack.dtype)
        to_store = da.store(scaled_stack, z, lock=False, compute=False)
        print(f'Writing mip with data to: {write_storepath}')
        to_store = progress(client.compute(to_store))
        to_store = client.gather(to_store)
        print()

    def build_zattrs(self, transformations):
        
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
            x = transformation['scale'][2]
            y = transformation['scale'][1]
            z = transformation['scale'][0]

            scale["coordinateTransformations"] = [{
                "scale": transformation['scale'],
                "type": "scale",
                }]
            
            scale["path"] = f'scale{mip}'
            
            datasets.append(scale)

        multiscales["datasets"] = datasets

        multiscales["type"] = (1, 2, 2)

        # Define down sampling methods for inclusion in zattrs
        description = '2x downsample of in up to 3 dimensions calculated using the local mean'
        details = 'stack_to_multiscale_ngff._builder_img_processing.local_mean_downsample'


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
        #colors = self.omero_dict['channels']['color']
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

        labels = self.channel
        channels = []
        for chn in range(1):
            new = {}
            new["active"] = True
            new["coefficient"] = 1
            new["color"] = colors[chn]
            new["family"] = "linear"
            new['inverted'] = False
            new['label'] = self.channel
            
            '''
            if self.dtype==np.dtype('uint8'):
                end = mx = 255
            elif self.dtype==np.dtype('uint16'):
                end = mx = 2**16 - 1
            else:
                end = mx = 1

            new['window'] = {
                "end": self.omero_dict['channels']['window'][chn]['end'] if self.omero_dict['channels']['window'] is not None else end,
                "max": self.omero_dict['channels']['window'][chn]['max'] if self.omero_dict['channels']['window'] is not None else mx,
                "min": self.omero_dict['channels']['window'][chn]['min'] if self.omero_dict['channels']['window'] is not None else 0,
                "start": self.omero_dict['channels']['window'][chn]['start'] if self.omero_dict['channels']['window'] is not None else 0
                }
            '''
            channels.append(new)
            
        omero['channels'] = channels
        
        omero['rdefs'] = {
            "defaultZ": 1,
            "model": "color"                  # "color" or "greyscale"
            }
        
        r.attrs['omero'] = omero

    def create_omezarrWATSON(self):
        print('Ome zarr manager setup')
        INPUT = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        OUTPUT = os.path.join(self.fileLocationManager.www, 'neuroglancer_data', 'C1T.ome.zarr')
        scales = (1, 1, 20.0, 10.4, 10.4)
        originalChunkSize = (1, 1, 1, 1024, 1024)
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