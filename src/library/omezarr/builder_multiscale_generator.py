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


    def write_transfer(self, client):
        print()
        start_time = timer()
        if os.path.exists(self.transfer_path):
            print(f'Image stack to zarr data already exists at {self.transfer_path}')                
            if self.debug:
                store = zarr.storage.NestedDirectoryStore(self.transfer_path)
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')
            return
        chunks = self.originalChunkSize
        print(f"Transferring data from image stack to zarr to {self.transfer_path}")
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
        elif self.ndim == 3:
            stack = da.moveaxis(stack, source=[3, 0], destination=[0, 1])
            stack = stack[None, ...]  # Add time dimension
        else:
            print(f'Unexpected sample.ndim={self.ndim} for stack {stack}')
            print(f'sample shape={self.img_shape} dtype={self.dtype}')
            print(f'stack shape={stack.shape} chunksize={stack.chunksize} dtype={stack.dtype}')
            print('This is not a 2D or 3D image stack, exiting')
            sys.exit(1)

        stack = stack.rechunk(chunks)  # Rechunk to original chunk size
        store = get_store_from_path(self.transfer_path)
        z = zarr.zeros(
            stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=self.compressor,
            dtype=stack.dtype,
        )

        if client is None:
            with ProgressBar():
                da.store(stack, z, lock=True, compute=True)
        else:
            to_store = da.store(stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

        volume = zarr.open(store, 'r')
        print(volume.info)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Transfer from TIF stack to zarr completed in {total_elapsed_time} seconds")

    def write_rechunk_transfer(self, client, chunks, input_path, write_storepath):
        print()
        start_time = timer()
        
        if not os.path.exists(input_path):
            print(f'Transferred data does not exist at {input_path} exiting')
            exit(1)
        
        if os.path.exists(write_storepath):
            print(f'Rechunked data exists at {write_storepath}')
            if self.debug:
                store = store = zarr.storage.NestedDirectoryStore(write_storepath)
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')
            return

        print(f"Building rechunked zarr store with {chunks=}:")
        print(f"\tfrom {input_path}")
        print(f"\tto {write_storepath}")

        stack = da.from_zarr(url=input_path)
        stack = stack.rechunk(chunks)  # Rechunk to original chunk size
        store = get_store_from_path(write_storepath)
        z = zarr.zeros(
            stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=self.compressor,
            dtype=stack.dtype,
        )

        if client is None:
            with ProgressBar():
                da.store(stack, z, lock=True, compute=True)
        else:
            to_store = da.store(stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

        volume = zarr.open(store, 'r')
        print(volume.info)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Rechunkin completed in {total_elapsed_time} seconds")


    def write_mips(self, mip, client):
        print()

        read_storepath = os.path.join(self.output, str(mip - 1))
        if not os.path.exists(read_storepath):
            print(f'Resolution {mip} does not exist at {read_storepath}')
            return

        write_storepath = os.path.join(self.output, str(mip))
        chunks = self.pyramidMap[mip]['chunk']
        if os.path.exists(write_storepath):
            print(f'Resolution {mip} exists at {write_storepath}')
            if self.debug:
                store = store = zarr.storage.NestedDirectoryStore(write_storepath)
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')
            return

        print(f"Building downsampled zarr store for resolution {mip}:")
        print(f"\tfrom {read_storepath}")
        print(f"\tto {write_storepath}")

        previous_stack = da.from_zarr(url=read_storepath)
        axis_scales = [1, 1, 1, 2, 2]
        axis_dict = {
            0: axis_scales[0],
            1: axis_scales[1],
            2: axis_scales[2],
            3: axis_scales[3],
            4: axis_scales[4],
        }
        scaled_stack = da.coarsen(mean_dtype, previous_stack, axis_dict, trim_excess=True)
        scaled_stack = scaled_stack.rechunk(chunks)
        print(f'\tNew store with shape={scaled_stack.shape} chunks={chunks}')

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
        print(f'\nCleaning up {self.scratch_space} and orphaned lock files')

        countKeyboardInterrupt = 0
        countException = 0
        while True:
            try:
                # Remove any existing files in the scratch_space
                files = glob.glob(os.path.join(self.scratch_space, "**/*"), recursive=True)
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

        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space)

