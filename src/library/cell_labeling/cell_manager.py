import os

import cv2
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from library.utilities.utilities_process import read_image, write_image




class CellManager:
    """
    This class is responsible for managing the cell labeling process, including loading and saving cell data, handling annotations, and interfacing with the database.
    """



    def create_roi(self, polygons_by_section):
        """
        Create a region of interest (ROI) based on the mask created from polygons.
        """
        # Implementation for creating ROI
        print("Creating ROI.")
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='aligned')
        if not os.path.exists(self.input):
            print(f'Input path missing: {self.input}')
            exit(1)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='axons')
        cropped_path = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='cropped')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(cropped_path, exist_ok=True)
        sorted_dict = dict(sorted(polygons_by_section.items()))
        for section_number, polygon in sorted_dict.items():
            tifname = f"{section_number:03d}.tif"
            filename = os.path.join(self.input, tifname)
            if os.path.exists(filename):
                image_outpath = os.path.join(self.output, tifname)
                cropped_outpath = os.path.join(cropped_path, tifname)
                if os.path.exists(image_outpath) and os.path.exists(cropped_outpath):
                    print(f'Skipping {tifname} in cropped and ROI, already exists.')
                    continue
                polygon = polygon.astype(np.int32)
                image_inpath = os.path.join(self.input, tifname)
                image = read_image(image_inpath)

                mask = np.zeros(image.shape, dtype=np.uint16)
                cv2.fillPoly(mask, [polygon], 65535)
                # Apply the mask using a bitwise AND operation
                # Everything outside the ROI becomes 0
                masked_image = cv2.bitwise_and(image, mask)
                x, y, w, h = cv2.boundingRect(polygon)
                cropped_image = masked_image[y:y+h, x:x+w]
                write_image(cropped_outpath, cropped_image, message=f'Error writing {cropped_outpath}')
                try:
                    axon_image, threshold = self.extract_axon_threshold(cropped_image)
                except Exception as e:
                    print(f'Could not find threshold for {tifname} ')
                    print(e)
                    axon_image = masked_image.copy()
                    threshold = 0
                axon_image = self.remove_white_objects(axon_image, threshold, min_size=10)
                masked_image[y:y+h, x:x+w] = axon_image
                write_image(image_outpath, masked_image, message=f'Error writing {image_outpath}')
                print(f'{threshold=}, Wrote: {cropped_outpath}')

    def add_empty_images(self):
        """
        Add empty images for sections that do not have any polygons.
        """
        print("Adding empty images for missing sections.")
        input_dir = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='aligned')
        output_dir = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='axons')
        os.makedirs(output_dir, exist_ok=True)

        # Get the list of all section numbers from the input directory
        all_sections = {int(os.path.splitext(f)[0]) for f in os.listdir(input_dir) if f.endswith('.tif')}
        # Get the list of section numbers that have polygons
        polygon_sections = set(self.polygons_by_section.keys())
        # Find the missing sections
        missing_sections = sorted(all_sections - polygon_sections)

        for section_number in tqdm(missing_sections, desc="Adding empty images", unit="section"):
            tifname = f"{section_number:03d}.tif"
            input_path = os.path.join(input_dir, tifname)
            output_path = os.path.join(output_dir, tifname)
            if os.path.exists(output_path):
                continue
            if os.path.exists(input_path):
                image = read_image(input_path)
                empty_image = np.zeros_like(image)
                write_image(output_path, empty_image, message=f'Error writing {output_path}')
                print(f'Added empty image for missing section: {tifname}')

    def create_roi_neuroglancer(self):
        """
        Create a region of interest (ROI) for Neuroglancer visualization.
        """
        # Implementation for creating ROI for Neuroglancer
        print("Creating ROI for Neuroglancer.")
        pass

    @staticmethod
    def remove_white_objects(img_16bit, thresh_value, min_size=10):
        # Set an appropriate intensity threshold (e.g., 50000 out of 65535, or use your own value)
        _, binary_mask = cv2.threshold(img_16bit, thresh_value, 1, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        
        # 3. Apply Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
        
        # Define the minimum area (in pixels) for an object to be kept
        
        # Create a mask for objects to keep
        keep_mask = np.zeros_like(binary_mask)
        
        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                keep_mask[labels == i] = 1
        
        # 4. Remove small objects from the original 16-bit image
        # Set pixels belonging to small objects to background intensity (e.g., 0 or local background)
        background_val = 0
        cleaned_img = img_16bit.copy()
        cleaned_img[(binary_mask == 1) & (keep_mask == 0)] = background_val
        return cleaned_img


    @staticmethod
    def extract_axon_threshold(img):
        """
        Extracts the axon separation threshold from a noisy image histogram using the maximum 
        value of the second derivative after the dominant background noise peak.
        
        Parameters:
        -----------
        data : numpy.ndarray
            The input image array (contains structural zeros, noise, and sparse high-intensity axons).
        intensity_range : tuple (min_val, max_val), optional
            The intensity window to look inside for the noise distribution. Helps isolate 
            the true noise floor if the histogram contains multiple irrelevant mini-peaks.
            
        Returns:
        --------
        float
            The calculated intensity threshold value separating axons from background noise.
        """
        foreground_data = img[img > 0]
        num_bins = 100
        bins = np.logspace(np.log10(foreground_data.min()), np.log10(foreground_data.max()), num_bins)
        counts, bin_edges = np.histogram(foreground_data, bins=bins)

        log_bin_centers = 0.5 * (np.log10(bin_edges[:-1]) + np.log10(bin_edges[1:]))
        bin_centers = 10 ** log_bin_centers
        log_counts = np.log10(counts + 1)

        # Smooth to get clear, continuous derivative curves
        smooth_log_counts = gaussian_filter1d(log_counts, sigma=2.0)
        dy_dx = np.gradient(smooth_log_counts, log_bin_centers)
        d2y_dx2 = np.gradient(dy_dx, log_bin_centers)

        # =====================================================================
        # FEATURE 1: Inflection point at the noise peak
        # =====================================================================
        # Step A: Locate the absolute maximum peak of the noise (usually at lower intensities)
        kde_x = np.linspace(foreground_data.min(), np.percentile(foreground_data, 90), 500)
        kde_y = gaussian_kde(foreground_data)(kde_x)

        # The maximum of the density curve is your noise peak
        noise_floor_limit = kde_x[np.argmax(kde_y)]        

        noise_indices = np.where(bin_centers < noise_floor_limit)[0]
        noise_peak_idx = noise_indices[np.argmax(smooth_log_counts[noise_indices])]
        #noise_peak_x = bin_centers[noise_peak_idx]

        # Step B: Find the inflection point on the descending slope of this peak (where d2y_dx2 crosses 0)
        # Look forward from the peak index
        zero_crossings = np.where(np.diff(np.sign(d2y_dx2)))[0]
        post_peak_crossings = zero_crossings[zero_crossings > noise_peak_idx]

        if len(post_peak_crossings) > 0:
            noise_inflection_idx = post_peak_crossings[0]  # First zero-crossing right after the peak
        else:
            noise_inflection_idx = noise_peak_idx + 5  # Fallback if no clean crossing is found

        axon_threshold = bin_centers[noise_inflection_idx]

        img[img < axon_threshold] = 0

        return img, float(axon_threshold)


