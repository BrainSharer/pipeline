"""Creates all histograms that displayed in the database portal
"""
import os
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2

from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_process import test_dir

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
            print(f"DEBUG: START HistogramMaker::make_histogram")
            workers = 1

        if self.downsample:
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            if not os.path.exists(self.input):
                print(f"Input path does not exist {self.input}")
                return
            self.masks = self.fileLocationManager.get_thumbnail_masked(channel=1) # hard code this to channel 1
            if not os.path.exists(self.masks):
                print(f"Mask path does not exist {self.masks}")
                return
            #files = self.sqlController.get_sections(self.animal, self.channel, self.debug)
            files, nfiles, *_ = test_dir(self.animal, self.input, self.section_count, downsample=True, same_size=False)
            if nfiles == 0:
                print("No sections in the database or folder")
            self.output = self.fileLocationManager.get_histogram(self.channel)
            os.makedirs(self.output, exist_ok=True)
            file_keys = []
            for i, file in enumerate(files):
                filename = str(i).zfill(3) + ".tif"
                input_path = os.path.join(self.input, filename)
                mask_path = os.path.join(self.masks, filename)
                output_path = os.path.join(
                    self.output, os.path.splitext(filename)[0] + ".png"
                )
                if not os.path.exists(input_path):
                    print("Input tif does not exist", input_path)
                    continue
                if os.path.exists(output_path):
                    continue
                file_keys.append(
                    [input_path, mask_path, self.channel, file, output_path]
                )

            self.run_commands_concurrently(make_single_histogram, file_keys, workers)


    def make_combined_histogram(self):
        """This method takes all tif files by channel and creates a histogram of the combined image space.
        
        :param animal: the prep_id of the animal we are working with
        :param channel: the channel {1,2,3}
        :return: nothing
        """

        if self.downsample:
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

    input_path, mask_path, channel, file, output_path = file_key

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

    #PREV. (PRIOR TO 27-SEP-2024) BELOW
    #REVISIONS:
    #-Combine filtering and flattening into one step: img[img > 0].ravel() instead of separate operations.
    #-Use ravel() instead of flatten() as it's generally faster (returns a view instead of a copy when possible).
    #-Use ax.hist() instead of plt.hist() for better control and to avoid creating multiple figures.
    #-Set the number of bins to the minimum of flat.max() and 1000 to prevent excessive memory usage for high bit-depth images.
    #-Enable log scale for y-axis by default, as it's often more informative for image histograms. (unclear if David/Beth wants this)
    #-Create figure and axes in one step with fig, ax = plt.subplots().
    #-Removed unnecessary del statements as Python's garbage collector will handle memory management.


    # img = read_image(input_path)
    # dtype = img.dtype
    # if img.dtype == np.uint8:
    #     end = 255
    # else:
    #     end = 65535

    # mask = read_image(mask_path)

    # try:
    #     img = cv2.bitwise_and(img, img, mask=mask)
    # except Exception as e:
    #     print(f"Could not apply mask {mask_path} to {input_path}")
    #     print(e)
    #     return

    # img = img[img > 0]
    # try:
    #     flat = img.flatten()
    # except:
    #     print(f"Could not flatten {input_path}")
    #     return
    
    # del img
    # del mask
    # fig = plt.figure()
    # plt.rcParams["figure.figsize"] = [10, 6]
    # plt.hist(flat, flat.max(), [0, end], color=COLORS[channel])
    # plt.style.use("ggplot")

    #plt.yscale("log")

    # plt.grid(axis="y", alpha=0.75)
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.title(f"{file.file_name} @{dtype}", fontsize=8)
    # plt.close()
    # fig.savefig(output_path, bbox_inches="tight")
    # return