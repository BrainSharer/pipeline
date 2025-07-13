# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:46:17 2023

@author: awatson
"""

'''
Store mix-in classes for utilities

These classes are designed to be inherited by the builder class (builder.py)
'''

import glob
import os
import sys
from time import sleep
import shutil
import time
from itertools import product
from timeit import default_timer as timer
import zarr
from dask.delayed import delayed
import dask.array as da
import dask
import skimage.io
from distributed import progress

# Project specific imports
from library.omezarr import utils
from library.utilities.dask_utilities import mean_dtype

class BuilderMultiscaleGenerator:


    def write_resolution_0(self, client):
        start_time = timer()
        resolution_0_path = os.path.join(self.output, '0')
        if os.path.exists(resolution_0_path):
            print(f'Resolution 0 already exists at {resolution_0_path}')                
            if self.debug:
                store = store = zarr.storage.NestedDirectoryStore(resolution_0_path)
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
        elif self.ndim == 3:
            stack = da.moveaxis(stack, source=[3, 0], destination=[0, 1])
            stack = stack[None, ...]  # Add time dimension
        else:
            print(f'Unexpected sample.ndim={self.ndim} for stack {stack}')
            print(f'sample shape={self.img_shape} dtype={self.dtype}')
            print(f'stack shape={stack.shape} chunksize={stack.chunksize} dtype={stack.dtype}')
            print('This is not a 2D or 3D image stack, exiting')
            sys.exit(1)

        print(f'Stack after reshaping type: {type(stack)} shape: {stack.shape} chunks: {stack.chunksize} dtype: {stack.dtype}')
        chunks = (1, 1, ) + self.pyramidMap[0]['chunk']        
        stack = stack.rechunk(chunks)  # Rechunk to original chunk size
        print(f'Stack after rechunking type: {type(stack)} shape: {stack.shape} chunks: {stack.chunksize} dtype: {stack.dtype}')
        store = self.get_store(0)
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
            to_store = da.store(stack, z, lock=True, compute=True)
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
        chunks = (1, 1, ) + self.pyramidMap[mip]['chunk'] # add extra dimenision at the beginning for time and channel
        print(f'chunks = {chunks}')
        scaled_stack = scaled_stack.rechunk(chunks)
        print(f'New store with shape={scaled_stack.shape} chunks={chunks}')

        store = self.get_store(mip)
        z = zarr.zeros(
            scaled_stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=self.compressor,
            dtype=scaled_stack.dtype,
        )

        if client is None:
            to_store = da.store(scaled_stack, z, lock=True, compute=True)
        else:
            to_store = da.store(scaled_stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)

    def write_resolutions(self, mip, client):

        # During creation of res 1, the min and max is calculated for res 0 if values
        # for omero window were not specified in the commandline
        # These values are added to zattrs omero:channels
        # out is a list of tuple (min,max,channel)

        resolution_path = os.path.join(self.output, str(mip))
        if os.path.exists(resolution_path):
            print(f'Resolution {mip} already exists at {resolution_path}')
            return

        minmax = False
        where_to_calculate_min_max = 2
        if len(self.pyramidMap) == 2 and mip == 1:
            minmax = True
        elif mip == len(self.pyramidMap) // where_to_calculate_min_max:
            minmax = True
        elif len(self.pyramidMap) // where_to_calculate_min_max == 0:
            minmax = True

        start_time = timer()
        results = self.down_samp_by_chunk_no_overlap(mip, client, minmax=minmax)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"Resolution {mip} completed in {total_elapsed_time} seconds")

        if minmax and self.omero_dict['channels']['window'] is None:
            self.min = []
            self.max = []
            for ch in range(self.channels):
                # Sort by channel
                tmp = [x[-1] for x in results if x[-1][-1] == ch]
                # Append vlues to list in channel order
                self.min.append( min([x[0] for x in tmp]) )
                self.max.append( max([x[1] for x in tmp]) )
            self.set_omero_window()

    ##############################################################################################################
    '''
    Functions below are newer downsample control for chunk-by-chunk method
    '''
    ##############################################################################################################

    def downsample_by_chunk(self, from_mip, to_mip, down_sample_ratio, from_slice, to_slice, minmax=False):

        '''
        Slices are for dims (t,c,z,y,x), downsample only works on 3 dims
        '''
        print(f'from_slice len={len(from_slice)} to_slice len={len(to_slice)}')

        dsamp_method = self.local_mean_downsample

        from_array = self.open_store(from_mip, mode='r')
        to_array = self.open_store(to_mip)
        data = from_array[from_slice]

        # Calculate min/max of input data
        if minmax:
            min, max = data.min(), data.max()

        #####data = data[0,0]
        data = dsamp_method(data,down_sample_ratio=down_sample_ratio)
        #####data = data[None, None, ...]
        to_array[to_slice] = data
        if minmax:
            return True, (min, max, from_slice[1].start)
        else:
            return True,

    def chunk_increase_to_limit_3d(self,image_shape,starting_chunks,chunk_limit_in_GB):

        # Determine chunk values to start at
        z = starting_chunks[0] if starting_chunks[0] < image_shape[0] else image_shape[0]
        y = starting_chunks[1] if starting_chunks[1] < image_shape[1] else image_shape[1]
        x = starting_chunks[2] if starting_chunks[2] < image_shape[2] else image_shape[2]

        previous_chunks = (z,y,x)
        current_chunks = (z,y,x)

        # Iterate 1 axis at a time increasing chunk size by increase_rate until the chunk_limit_in_MB is reached
        while True:
            if utils.get_size_GB(current_chunks,self.dtype) >= chunk_limit_in_GB:
                current_chunks = previous_chunks
                break
            else:
                previous_chunks = current_chunks

            if z < image_shape[0]:
                z = z + starting_chunks[0] if starting_chunks[0] < image_shape[0] else image_shape[0]
            else:
                break

            current_chunks = (z,y,x)

        return current_chunks

    def chunk_slice_generator_for_downsample(self, from_array, to_array, down_sample_ratio=(2, 2, 2), length=False):
        """
        Generate slice for each chunk in array for shape and chunksize
        Also generate slices for each chunk for an array of 2x size in each dim.

        from_array:
            array-like obj with .chunksize or .chunks and .shape
            OR
            tuple of tuples ((),()) where index [0] == shape, index [1] == chunks
        to_array:
            array-like obj with .chunksize or .chunks and .shape
            OR
            tuple of tuples ((),()) where index [0] == shape, index [1] == chunks

        Assume dims are (t,c,z,y,x)
        downsample ratio is a tuple if int (z,y,x)
        """
        if isinstance(to_array, tuple):
            chunksize = to_array[1]
            shape = to_array[0]
        else:
            try:
                chunksize = to_array.chunksize
            except:
                chunksize = to_array.chunks
            shape = to_array.shape

        if isinstance(from_array, tuple):
            from_chunksize = from_array[1]
            from_shape = from_array[0]
        else:
            try:
                from_chunksize = from_array.chunksize
            except:
                from_chunksize = from_array.chunks
            from_shape = from_array.shape

        # Chunks are calculated for to_array
        ##### Change
        optimum_chunks = self.chunk_increase_to_limit_3d(shape[1:], chunksize[1:],
                                                         self.res_chunk_limit_GB)

        chunksize = list(chunksize[:2]) + list(optimum_chunks)
        # print(f'Chunksize for processing downsampled array: {chunksize}')
        # chunk_multiply = 2
        # chunksize = list(chunksize[:2]) + [x*chunk_multiply for x in chunksize[-3:]]

        chunks = [x // y for x, y in zip(shape, chunksize)]
        chunk_mod = [x % y > 0 for x, y in zip(shape, chunksize)]
        chunks = [x + 1 if y else x for x, y in zip(chunks, chunk_mod)]

        reverse_chunks = chunks[::-1]
        reverse_chunksize = chunksize[::-1]
        reverse_shape = shape[::-1]
        reverse_from_shape = from_shape[::-1]

        # Tuple of 3 values
        reverse_downsample_ratio = down_sample_ratio[::-1]

        if length:
            yield from product(*map(range, reverse_chunks))
        else:
            for a in product(*map(range, reverse_chunks)):
                start = [x * y for x, y in zip(a, reverse_chunksize)]
                stop = [(x + 1) * y for x, y in zip(a, reverse_chunksize)]
                stop = [x if x < y else y for x, y in zip(stop, reverse_shape)]

                to_slices = [slice(x, y) for x, y in zip(start, stop)]
                # print(to_slices)

                start = [x * y for x,y in zip(start[:-2],reverse_downsample_ratio)] + start[-2:]
                stop = [x * y for x,y in zip(stop[:-2],reverse_downsample_ratio)] + stop[-2:]
                stop = [x if x < y else y for x, y in zip(stop, reverse_from_shape)]
                from_slices = [slice(x, y) for x, y in zip(start, stop)]
                # print(slices[::-1])
                yield tuple(from_slices[::-1]), tuple(to_slices[::-1])

    @staticmethod
    def compute_govenor(processing, num_at_once=70, complete=False, keep_results=False):
        results = []
        while len(processing) >= num_at_once:
            still_running = [x.status != 'finished' for x in processing]

            # Partition completed from processing
            if keep_results:
                results_tmp = [x for x,y in zip(processing,still_running) if not y]
                results = results + results_tmp
                del results_tmp
            processing = [x for x,y in zip(processing,still_running) if y]

            time.sleep(0.1)

        if complete:
            while len(processing) > 0:
                print('                                                                                      ',
                      end='\r')
                print(f'{len(processing)} remaining', end='\r')
                still_running = [x.status != 'finished' for x in processing]

                # Partition completed from processing
                if keep_results:
                    results_tmp = [x for x, y in zip(processing, still_running) if not y]
                    results = results + results_tmp
                    del results_tmp
                processing = [x for x, y in zip(processing, still_running) if y]

                time.sleep(0.1)
        return results, processing

    def down_samp_by_chunk_no_overlap(self, mip, client, minmax=False):

        new_array_store = self.get_store(mip)

        ##### Changes
        #####new_shape = (self.TimePoints, self.channels, *self.pyramidMap[mip]['shape'])
        #####new_chunks = (1, 1, *self.pyramidMap[mip]['chunk'])
        new_shape = (self.channels, *self.pyramidMap[mip]['shape'])
        new_chunks = (1, *self.pyramidMap[mip]['chunk'])

        new_array = zarr.zeros(new_shape, chunks=new_chunks, store=new_array_store, overwrite=True,
                               compressor=self.compressor, dtype=self.dtype)

        ##### Changes
        """
        from_array_shape_chunks = (
            (self.TimePoints, self.channels, *self.pyramidMap[mip - 1]["shape"]),
            (1, 1, *self.pyramidMap[mip - 1]["chunk"]),)

        to_array_shape_chunks = (
            (self.TimePoints, self.channels, *self.pyramidMap[mip]["shape"]),
            (1, 1, *self.pyramidMap[mip]["chunk"]),)
        """
        from_array_shape_chunks = (
            (self.channels, *self.pyramidMap[mip - 1]["shape"]),
            (1, *self.pyramidMap[mip - 1]["chunk"]),)

        to_array_shape_chunks = (
            (self.channels, *self.pyramidMap[mip]["shape"]),
            (1, *self.pyramidMap[mip]["chunk"]),)

        down_sample_ratio = self.pyramidMap[mip]['downsample']

        slices = self.chunk_slice_generator_for_downsample(from_array_shape_chunks, to_array_shape_chunks,
                                                      down_sample_ratio=down_sample_ratio, length=True)
        total_slices = len(tuple(slices))
        slices = self.chunk_slice_generator_for_downsample(from_array_shape_chunks, to_array_shape_chunks,
                                                      down_sample_ratio=down_sample_ratio, length=False)
        num = 0
        processing = []
        start = time.time()

        final_results = []
        for from_slice, to_slice in slices:
            # print(from_slice)
            # print(to_slice)
            num+=1
            if num % 100 == 1:
                mins_per_chunk = (time.time() - start) / num / 60
                mins_remaining = (total_slices - num) * mins_per_chunk
                mins_remaining = round(mins_remaining, 2)
                print(f'Computing chunks {num} of {total_slices} : {mins_remaining} mins remaining', end='\r')
            else:
                print(f'Computing chunks {num} of {total_slices} : {mins_remaining} mins remaining', end='\r')
            if client is not None:
                tmp = delayed(self.downsample_by_chunk)(mip-1, mip, down_sample_ratio, from_slice, to_slice, minmax=minmax)
                tmp = client.compute(tmp)
                processing.append(tmp)
            else:
                tmp = self.downsample_by_chunk(mip-1, mip, down_sample_ratio, from_slice, to_slice, minmax=minmax)
                if minmax:
                    final_results.append(tmp)

            del tmp
            results, processing = self.compute_govenor(processing, num_at_once=round(self.cpu_cores*4), complete=False, keep_results=minmax)

            final_results = final_results + results

        results, processing = self.compute_govenor(processing, num_at_once=round(self.cpu_cores*4), complete=True, keep_results=minmax)
        final_results = final_results + results

        if minmax:
            final_results = client.gather(final_results)
            # return final_results

        return final_results

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
        sleep(1) # give the system time to finish writing
        
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

