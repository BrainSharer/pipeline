import glob
import os
import sys
import shutil
from timeit import default_timer as timer
import zarr
import dask.array as da
import dask
from dask.diagnostics import ProgressBar

import skimage.io
from distributed import progress

# Project specific imports
from library.utilities.dask_utilities import get_store_from_path, mean_dtype

class BuilderMultiscaleGenerator:


    def write_resolution_0(self, client):
        start_time = timer()
        resolution_0_path = os.path.join(self.output, '0')
        if os.path.exists(resolution_0_path):
            print(f'Resolution 0 already exists at {resolution_0_path}')                
            if self.debug:
                store = zarr.storage.NestedDirectoryStore(resolution_0_path)
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')
            return

        print(f"Building zarr store for resolution 0 at {resolution_0_path}")
        imread = dask.delayed(skimage.io.imread, pure=True)  # Lazy version of imread
        lazy_images = [imread(path) for path in sorted(self.files)]   # Lazily evaluate imread on each path

        arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                                dtype=self.dtype,   # for every lazy value
                                shape=self.img_shape)
                for lazy_image in lazy_images]

        stack = da.stack(arrays, axis=0)
        print(f'Stack after stacking type: {type(stack)} shape: {stack.shape} chunks: {stack.chunksize} dtype: {stack.dtype}')

        if self.ndim == 2:
            stack = stack[None, None, ...]  # Add time and channel dimensions
            chunks = self.pyramidMap[0]['chunk']        
        elif self.ndim == 3:
            stack = da.moveaxis(stack, source=[3, 0], destination=[0, 1])
            stack = stack[None, ...]  # Add time dimension
            chunks = self.pyramidMap[0]['chunk']        
        else:
            print(f'Unexpected sample.ndim={self.ndim} for stack {stack}')
            print(f'sample shape={self.img_shape} dtype={self.dtype}')
            print(f'stack shape={stack.shape} chunksize={stack.chunksize} dtype={stack.dtype}')
            print('This is not a 2D or 3D image stack, exiting')
            sys.exit(1)

        print(f'Stack after reshaping and rechunking type: {type(stack)} shape: {stack.shape} chunks: {stack.chunksize} dtype: {stack.dtype}')
        stack = stack.rechunk(chunks)  # Rechunk to original chunk size
        print(f'Stack after rechunking type: {type(stack)} shape: {stack.shape} chunks: {stack.chunksize} dtype: {stack.dtype}')
        store = get_store_from_path(resolution_0_path)
        z = zarr.zeros(
            stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=self.compressor,
            dtype=stack.dtype,
        )
        print('Stacked z info')
        print(z.info)

        if client is None:
            with ProgressBar():
                da.store(stack, z, lock=True, compute=True)
        else:
            to_store = da.store(stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Resolution 0 completed in {total_elapsed_time} seconds")



    def write_mips(self, mip, client):
        print()
        read_storepath = os.path.join(self.output, str(mip-1))
        if os.path.exists(read_storepath):
            print(f'Resolution {mip-1} exists at {read_storepath} loading ...')
        else:
            print(f'Resolution {mip-1} does not exist at {read_storepath}')
            exit(1)

        write_storepath = os.path.join(self.output, str(mip))
        if os.path.exists(write_storepath):
            print(f'Resolution {mip} exists at {write_storepath}')
            if self.debug:
                store = store = zarr.storage.NestedDirectoryStore(write_storepath)
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')
            return

        print(f'Creating resolution {mip} from {read_storepath}')
        previous_stack = da.from_zarr(url=read_storepath)
        print(f'Creating new store from previous shape={previous_stack.shape} chunks={previous_stack.chunksize}')
        axis_scales = [1, 1, 1, 2, 2]
        axis_dict = {
            0: axis_scales[0],
            1: axis_scales[1],
            2: axis_scales[2],
            3: axis_scales[3],
            4: axis_scales[4],
        }
        scaled_stack = da.coarsen(mean_dtype, previous_stack, axis_dict, trim_excess=True)
        chunks = self.pyramidMap[mip]['chunk'] # add extra dimenision at the beginning for time and channel
        print(f'chunks = {chunks}')
        scaled_stack = scaled_stack.rechunk(chunks)
        print(f'New store with shape={scaled_stack.shape} chunks={chunks}')

        store = get_store_from_path(write_storepath)
        z = zarr.zeros(
            scaled_stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=self.compressor,
            dtype=scaled_stack.dtype,
        )

        if client is None:
            with ProgressBar():
                da.store(scaled_stack, z, lock=True, compute=True)
        else:
            to_store = da.store(scaled_stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)


    def cleanup(self):
        """
            Cleans up the temporary directory and removes orphaned lock files.

            This method removes all files and directories in the temporary directory
            and removes any .lock files in the output directory.

            Raises:
                KeyboardInterrupt: If the cleanup process is interrupted by the user.
                Exception: If an error occurs during the cleanup process.
            """
        print(f'\nCleaning up {self.tmp_dir} and orphaned lock files')
        
        countKeyboardInterrupt = 0
        countException = 0
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

