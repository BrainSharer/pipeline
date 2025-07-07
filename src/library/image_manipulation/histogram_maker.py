"""Creates all histograms that displayed in the database portal
"""
import os
import inspect
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2
import tifffile
from pathlib import Path

from library.image_manipulation.image_manager import ImageManager
from library.controller.sections_controller import SectionsController
from library.utilities.utilities_process import test_dir, SCALING_FACTOR, DOWNSCALING_FACTOR

COLORS = {1: "b", 2: "r", 3: "g"}


class HistogramMaker:
    """Includes methods to generate single histograms per tiff file and a combined histogram for entire stack
    """

    def make_histogram(self):
        """
        This method creates an individual histogram for each tif file by channel.
        
        :param animal: the prep id of the animal
        :param channel: the channel of the stack to process  {1,2,3}
        :returns: nothing
        """
        workers = self.get_nworkers()

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")
            workers = 1

        if self.downsample:
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            
            section_count = self.sqlController.get_section_count(self.animal)

            if section_count == 0:
                if os.path.exists(self.input):
                    section_count = len(os.listdir(self.input))
            section_count = section_count
            
            if not os.path.exists(self.input):
                print(f"Input path does not exist {self.input}")
                return
            self.masks = self.fileLocationManager.get_thumbnail_masked(channel=1) # hard code this to channel 1
            if not os.path.exists(self.masks):
                print(f"Mask path does not exist {self.masks}")
                return
            files, nfiles, *_ = test_dir(self.animal, self.input, section_count, downsample=True, same_size=False)
            if nfiles == 0:
                print("No sections in the database or folder")

            self.output = self.fileLocationManager.get_histogram(self.channel)
            os.makedirs(self.output, exist_ok=True)
            
            file_keys = []
            for i, file in enumerate(files):
                filename = str(i).zfill(3) + ".tif"
                input_path = os.path.join(self.input, filename)
                real_path = os.path.realpath(input_path) #FOLLOWS LINK TO ORG. FILENAME
                real_filename = os.path.basename(real_path)
                
                mask_path = os.path.join(self.masks, filename)
                output_path = os.path.join(
                    self.output, os.path.splitext(real_filename)[0] + ".png"
                )

                if not os.path.exists(real_path):
                    print("Input tif does not exist", real_path)
                    continue
                if os.path.exists(output_path):
                    continue
            
                file_keys.append(
                    [real_path, mask_path, self.channel, file, output_path, self.debug]
                )

            self.run_commands_concurrently(make_single_histogram, file_keys, workers)


    def make_combined_histogram(self):
        """This method takes all tif files by channel and creates a histogram of the combined image space.
        
        :param animal: the prep_id of the animal we are working with
        :param channel: the channel {1,2,3}
        :return: nothing
        """

        if not self.downsample:
            print('No histograms for full resolution images')
            return

        self.input = self.fileLocationManager.get_thumbnail(self.channel)
        if not os.path.exists(self.input):
            print(f"Input path does not exist {self.input}")
            return
        self.masks = self.fileLocationManager.get_thumbnail_masked(channel=1) #hard code this to channel 1
        if not os.path.exists(self.masks):
            print(f"Mask path does not exist {self.masks}")
            return

        self.output = self.fileLocationManager.get_histogram(self.channel)
        image_manager = ImageManager(self.input)
        files = image_manager.files
        dtype = image_manager.dtype
        lfiles = len(files)
        os.makedirs(self.output, exist_ok=True)
        hist_dict = Counter({})
        outfile = f"{self.animal}.png"
        outpath = os.path.join(self.output, outfile)
        if os.path.exists(outpath):
            return
        for file in files:
            file = os.path.basename(file)
            input_path = os.path.join(self.input, file)
            mask_path = os.path.join(self.masks, file)
            try:
                img = io.imread(input_path)
            except:
                print(f"Could not read {input_path}")
                lfiles -= 1
                continue
            try:
                mask = io.imread(mask_path)
            except Exception as e:
                print(f"ERROR WITH {e}")
                break
            #img = img[img > 0]                
            img = cv2.bitwise_and(img, img, mask=mask)

            try:
                flat = img.flatten()
                del img
            except:
                print(f"Could not flatten file {input_path}")
                lfiles -= 1
                continue
            try:
                img_counts = np.bincount(flat)
            except:
                print(f"Could not create counts {input_path}")
                lfiles -= 1
                continue
            try:
                img_dict = Counter(dict(zip(np.unique(flat), img_counts[img_counts.nonzero()])))
            except:
                print(f"Could not create counter {input_path}")
                lfiles -= 1
                continue
            try:
                hist_dict = hist_dict + img_dict
            except:
                print(f"Could not add files {input_path}")
                lfiles -= 1
                continue
        
        if lfiles < 10:
            return
        hist_dict = dict(hist_dict)
        hist_values = [i / lfiles for i in hist_dict.values()]
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = [10, 6]
        plt.bar(list(hist_dict.keys()), hist_values, color=COLORS[self.channel])
        plt.yscale("log")
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value")
        plt.xlim(0, 40000)
        #plt.ylim(0, 4000)
        plt.ylabel("Frequency")
        plt.title(f"{self.animal} channel {self.channel} @{dtype}bit with {lfiles} tif files", fontsize=8)
        fig.savefig(outpath, bbox_inches="tight")

def make_single_histogram(file_key: tuple[str, str, str, str, str]) -> None:
    """Makes a histogram for a single image file
    
    :param file_key: tuple of input_path, mask_path, channel, file, output_path
    """

    input_path, mask_path, channel, file, output_path, debug = file_key
    
    # Read image and mask
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Could not read image {input_path} or mask {mask_path}")
        return

    # Apply mask
    try:
        img = cv2.bitwise_and(img, img, mask=mask)
    except Exception as e:
        print(f"Could not apply mask {mask_path} to {input_path}: {e}")

        if debug:
            print(f"Image dimensions (height, width): {img.shape}")
            print(f"Mask dimensions (height, width): {mask.shape}")

        return
    
    # Filter out zero values and flatten
    flat = img[img > 0].ravel()

    if flat.size == 0:
        print(f"No non-zero pixels in masked image {input_path}")
        return

    # Determine histogram range
    dtype = img.dtype
    end = 255 if dtype == np.uint8 else 65535
    
    # Create and save histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(flat, bins=min(flat.max(), 1000), range=(0, end), color=COLORS[channel])
    ax.set_yscale('log')
    ax.grid(axis="y", alpha=0.75)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{file} @{dtype}", fontsize=8)
    plt.style.use("ggplot")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

##################################################
# WIP: SONG-MAO (3-JUL-2025)
##################################################
def make_histogram_full_RAM(img: np.ndarray, output_path_histogram: Path, mask_path: Path, debug: bool = False) -> None:
    '''Generate histogram for a single image [full resolution]'''

    output_path = Path(output_path_histogram) if isinstance(output_path_histogram, str) else output_path_histogram
    mask_path = Path(mask_path) if isinstance(mask_path, str) else mask_path

    if debug:
        print(f"DEBUG: Starting histogram generation for directory: {output_path.stem}")
        print(f"DEBUG: Processing: {output_path}")

    try:
        # Validate output path
        if not output_path.name.endswith('.png'):
            raise ValueError(f"Output must be PNG file, got {output_path.name}")
        if output_path.exists() and output_path.is_dir():
            raise ValueError(f"Path exists as directory: {output_path}")
        
        # Downsample if needed
        if SCALING_FACTOR > 1:
            new_dims = (int(img.shape[1] * DOWNSCALING_FACTOR), 
                       int(img.shape[0] * DOWNSCALING_FACTOR))
            img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask {mask_path}")
        
        # Resize IMAGE to match MASK dimensions if needed
        if img.shape[:2] != mask.shape[:2]:
            if debug:
                print(f"Resizing image from {img.shape[:2]} to match mask {mask.shape[:2]}")
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]), 
                           interpolation=cv2.INTER_AREA)
            
            if SCALING_FACTOR > 1:
                new_dims = (int(mask.shape[1] * DOWNSCALING_FACTOR),
                           int(mask.shape[0] * DOWNSCALING_FACTOR))
                img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, new_dims, interpolation=cv2.INTER_NEAREST)

        # Apply mask and get intensities
        masked = img[mask > 0]
        if masked.size == 0:
            print(f"No masked pixels in {output_path.stem}")
            return
        
        # Create and save histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        dtype = img.dtype
        max_val = (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) 
                  else np.percentile(masked, 99.9))
        
        ax.hist(masked, bins=min(256, len(np.unique(masked))), 
               range=(0, max_val), 
               color='blue')
        ax.set_yscale('log')
        ax.set_title(f"{output_path.stem}\n{SCALING_FACTOR}x downsampled")
        fig.savefig(str(output_path), bbox_inches='tight', dpi=150)  # str() for matplotlib
        plt.close(fig)

        if debug:
            print(f"DEBUG: Successfully saved histogram to {output_path}")
    except Exception as e:
        print(f"Error processing {output_path.stem}: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()


#POSSIBLY DEPRECATED BY make_histogram_full_RAM 
def make_single_histogram_full(file_key: tuple[str, str, str, str, str]) -> None:
    """Makes a histogram for a single image file [full resolution]
    
    :param file_key: tuple of input_path, mask_path, channel, file, output_path
    """
    input_path, mask_path, channel, file, output_path, debug = file_key

    try:
        # 1. Read and downsample main image
        with tifffile.TiffFile(input_path) as tif:
            img = tif.asarray()
            if SCALING_FACTOR > 1:
                new_dims = (int(img.shape[1] * DOWNSCALING_FACTOR), 
                          int(img.shape[0] * DOWNSCALING_FACTOR))
                img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)

        # 2. Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask {mask_path}")

        # 3. Resize mask if dimensions don't match
        if img.shape[:2] != mask.shape[:2]:
            if debug:
                print(f"Adjusting mask dimensions from {mask.shape} to {img.shape[:2]}")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
        # 4. Apply mask and get intensities
        masked = img[mask > 0]
        if masked.size == 0:
            print(f"No masked pixels in {input_path}")
            return
        
        # 5. Create and save histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        dtype = img.dtype
        max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.percentile(masked, 99.9)
        
        ax.hist(masked, bins=min(256, len(np.unique(masked))), 
               range=(0, max_val), 
               color=COLORS.get(channel, 'blue'))
        ax.set_yscale('log')
        ax.set_title(f"{Path(file).stem} \n{SCALING_FACTOR}x downsampled")
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()


def make_combined_histogram_full(image_manager, OUTPUT_DIR, animal, mask_path) -> None:
    #######################################
    #COMBINED HISTOGRAM - FULL RESOLUTION
    #######################################
    files = image_manager.files
    dtype = image_manager.dtype
    lfiles = len(files)
    hist_dict = Counter({})
    outfile = f"{animal}.png"
    outpath = Path(OUTPUT_DIR, outfile)
    print(f"DEBUG: Creating combined histogram from {len(files)} files")
    processed_files = 0

    for file in files:
        # print(f"Processing {file}")
        mask_file = Path(mask_path, Path(file).name)
        
        try:
            # 1. Read and downsample main image
            with tifffile.TiffFile(file) as tif:
                img = tif.asarray()
                if SCALING_FACTOR > 1:
                    new_dims = (int(img.shape[1] * DOWNSCALING_FACTOR), 
                               int(img.shape[0] * DOWNSCALING_FACTOR))
                    img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
            
            # 2. Read and validate mask
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_file}")
                continue
                
            # 3. Resize mask if needed
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Apply mask
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            del img  # Free memory
            
            # Flatten and count values
            flat = masked_img.flatten()
            del masked_img
            nonzero_values = flat[flat != 0]  # Only count non-zero values
            
            # Update histogram counts
            img_counts = np.bincount(nonzero_values)
            nonzero_indices = np.nonzero(img_counts)[0]
            img_dict = Counter(dict(zip(nonzero_indices, img_counts[nonzero_indices])))
            hist_dict += img_dict
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Only create histogram if we processed any files
    if processed_files > 0:
        print(f"Processed {processed_files} files successfully")
        
        # Convert to normalized values
        hist_items = sorted(hist_dict.items())
        keys = [k for k, v in hist_items]
        values = [v / processed_files for k, v in hist_items]
        
        # Create plot
        fig = plt.figure(figsize=(10, 6))
        plt.bar(keys, values, color=COLORS[1])
        plt.yscale("log")
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value")
        plt.xlim(0, 40000)
        plt.ylabel("Normalized Frequency")
        plt.title(f"{animal} @{dtype}bit with {processed_files} tif files", fontsize=8)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Warning: No files were processed successfully")