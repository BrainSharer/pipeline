import os
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import cv2
from utilities.utilities_process import test_dir, read_image

#from lib.pipeline_utilities import read_image  #MOVED TO utilities_process AS OF 4-NOV-2022


COLORS = {1: "b", 2: "r", 3: "g"}


class HistogramMaker:
    '''
    Includes methods to generate single histograms per tiff file and a combined histogram for entire stack

    Methods
    -------
    make_histogram()
    make_combined_histogram()

    '''

    def make_histogram(self):
        """
        This method creates an individual histogram for each tif file by channel.
        Args:
            animal: the prep id of the animal
            channel: the channel of the stack to process  {1,2,3}
        Returns:
            nothing
        """
        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            MASK_INPUT = self.fileLocationManager.thumbnail_masked
            files = self.sqlController.get_sections(self.animal, self.channel)
            test_dir(
                self.animal, INPUT, self.section_count, downsample=True, same_size=False
            )
            if len(files) == 0:
                print(" No sections in the database")
            OUTPUT = self.fileLocationManager.get_histogram(self.channel)
            os.makedirs(OUTPUT, exist_ok=True)
            self.sqlController.set_task_for_step(
                self.animal, True, self.channel, "HISTOGRAM"
            )
            file_keys = []
            for i, file in enumerate(files):
                filename = str(i).zfill(3) + ".tif"
                input_path = os.path.join(INPUT, filename)
                mask_path = os.path.join(MASK_INPUT, filename)
                output_path = os.path.join(
                    OUTPUT, os.path.splitext(file.file_name)[0] + ".png"
                )
                if not os.path.exists(input_path):
                    print("Input tif does not exist", input_path)
                    continue
                if os.path.exists(output_path):
                    continue
                file_keys.append(
                    [input_path, mask_path, self.channel, file, output_path]
                )
            count_physical_files = len(np.unique([i.file_name for i in files]))
            self.fileLogger.logevent(f"UNIQUE PHYSICAL FILE COUNT: {count_physical_files}")
            self.fileLogger.logevent(f"SECTION COUNT IN DATABASE: {len(files)}")

            workers = self.get_nworkers()
            self.run_commands_concurrently(make_single_histogram, file_keys, workers)


    def make_combined_histogram(self):
        """
        This method takes all tif files by channel and creates a histogram of the combined image space.
        :param animal: the prep_id of the animal we are working with
        :param channel: the channel {1,2,3}
        :return: nothing
        """
        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            MASK_INPUT = self.fileLocationManager.thumbnail_masked
            OUTPUT = self.fileLocationManager.get_histogram(self.channel)
            self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
            files = os.listdir(INPUT)
            files = [os.path.basename(i) for i in files]
            lfiles = len(files)
            self.fileLogger.logevent(f"CURRENT FILE COUNT: {lfiles}")
            self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            os.makedirs(OUTPUT, exist_ok=True)
            # files = os.listdir(INPUT) #deprecated 28-jun-2022
            hist_dict = Counter({})
            outfile = f"{self.animal}.png"
            outpath = os.path.join(OUTPUT, outfile)
            if os.path.exists(outpath):
                return
            midindex = lfiles // 2
            midfilepath = os.path.join(INPUT, files[midindex])
            img = io.imread(midfilepath)
            bits = img.dtype
            del img
            for file in files:
                input_path = os.path.join(INPUT, file)
                mask_path = os.path.join(MASK_INPUT, file)
                try:
                    img = io.imread(input_path)
                except:
                    self.fileLogger.logevent(f"Could not read {input_path}")
                    lfiles -= 1
                    continue
                try:
                    mask = io.imread(mask_path)
                    img = cv2.bitwise_and(img, img, mask=mask)
                except:
                    self.fileLogger.logevent(
                        f"ERROR WITH FILE OR DIMENSIONS: {mask_path}{mask.shape}{img.shape}"
                    )
                    break
                try:
                    flat = img.flatten()
                    del img
                except:
                    self.fileLogger.logevent(f"Could not flatten file {input_path}")
                    lfiles -= 1
                    continue
                try:
                    img_counts = np.bincount(flat)
                except:
                    self.fileLogger.logevent(f"Could not create counts {input_path}")
                    lfiles -= 1
                    continue
                try:
                    img_dict = Counter(
                        dict(zip(np.unique(flat), img_counts[img_counts.nonzero()]))
                    )
                except:
                    self.fileLogger.logevent(f"Could not create counter {input_path}")
                    lfiles -= 1
                    continue
                try:
                    hist_dict = hist_dict + img_dict
                except:
                    self.fileLogger.logevent(f"Could not add files {input_path}")
                    lfiles -= 1
                    continue
            if lfiles > 10:
                hist_dict = dict(hist_dict)
                hist_values = [i / lfiles for i in hist_dict.values()]
                fig = plt.figure()
                plt.rcParams["figure.figsize"] = [10, 6]
                plt.bar(list(hist_dict.keys()), hist_values, color=COLORS[self.channel])
                plt.yscale("log")
                plt.grid(axis="y", alpha=0.75)
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.title(
                    f"{self.animal} channel {self.channel} @{bits}bit with {lfiles} tif files"
                )
                fig.savefig(outpath, bbox_inches="tight")


def make_single_histogram(file_key):
    input_path, mask_path, channel, file, output_path = file_key
    img = read_image(input_path)
    mask = read_image(mask_path)
    img = cv2.bitwise_and(img, img, mask=mask)
    if img.shape[0] * img.shape[1] > 1000000000:
        scale = 1 / float(2)
        img = img[:: int(1.0 / scale), :: int(1.0 / scale)]
    try:
        flat = img.flatten()
    except:
        print(f"Could not flatten {input_path}")
        return
    del img
    del mask
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.hist(flat, flat.max(), [0, 10000], color=COLORS[channel])
    plt.style.use("ggplot")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"{file.file_name} @16bit")
    plt.close()
    fig.savefig(output_path, bbox_inches="tight")
    return
