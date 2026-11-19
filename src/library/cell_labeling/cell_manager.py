from datetime import datetime
import os

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from tables import Column
from tqdm import tqdm

from library.database_model.brain_region import CellData, RegionOfInterest
from library.utilities.utilities_process import is_non_empty_dir, read_image, write_image




class CellManager:
    """
    This class is responsible for managing the cell labeling process, including loading and saving cell data, handling annotations, and interfacing with the database.
    """



    def create_roi(self, polygons_by_section: dict):
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
                if os.path.exists(image_outpath):
                    print(f'Skipping {tifname} ROI as it already exists.')
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

    def create_roi_row(self, polygons_by_section: dict, volume_name: str):
        """
        Create a row in the database for the region of interest (ROI) based on the given section number and polygon.
        """
        # Implementation for creating a database row for ROI
        xy_resolution = self.sqlController.scan_run.resolution

        sorted_dict = dict(sorted(polygons_by_section.items()))
        for section_number, polygon in sorted_dict.items():
            area_px = cv2.contourArea(polygon.astype(np.int32))
            area_um2 = round(area_px * xy_resolution**2, 2)  # Convert area to square microns
            roi_row = self.sqlController.row_exists({"FK_state_id": self.state_id, "layer_name": self.layer_name, "section": section_number}, RegionOfInterest)
            if roi_row is not None:
                self.sqlController.update_row(RegionOfInterest, roi_row.id, {"area": area_um2})
            else:
                roi_row = RegionOfInterest(FK_state_id=self.state_id, FK_prep_id=self.animal, 
                                    layer_name=self.layer_name, volume_name=volume_name, section=section_number, 
                                    area=area_um2, active=1, created=datetime.now())
                self.sqlController.add_row(roi_row)

        # now update the COM for each ROI based on the contours
        input_dir = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='axons')
        infiles = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
        for infile in tqdm(infiles[0:26], desc="Updating ROI COM", unit="image", disable=self.debug):
            section_number = int(os.path.splitext(infile)[0])
            image_path = os.path.join(input_dir, infile)
            image = read_image(image_path)
            search_dict = {"FK_state_id": self.state_id, "layer_name": self.layer_name, "section": section_number}
            roi_row = self.sqlController.row_exists(search_dict, RegionOfInterest)
            if roi_row is None:
                continue
            # Calculate the center of mass (COM) for the ROI
            M = cv2.moments(image)
            if M["m00"] != 0:
                com_x = round(M["m10"] / M["m00"] * xy_resolution, 2)
                com_y = round(M["m01"] / M["m00"] * xy_resolution, 2)
                # Update the ROI row with the calculated COM
                self.sqlController.update_row(RegionOfInterest, roi_row.id, {"com_x": com_x, "com_y": com_y})


    def add_empty_images(self, polygons_by_section: dict):
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
        polygon_sections = set(polygons_by_section.keys())
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

    def setup_roi_neuroglancer(self) -> bool:
        """
        Create a region of interest (ROI) for Neuroglancer visualization.
        """
        # Implementation for creating ROI for Neuroglancer
        print("Creating ROI for Neuroglancer.")
        isFinished = False

        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='axons')
        self.output = os.path.join(self.fileLocationManager.neuroglancer_data, 'axons')
        self.rechunkme_path = os.path.join(self.fileLocationManager.neuroglancer_data, 'axons_rechunkme')
        self.progress_dir = os.path.join(self.fileLocationManager.neuroglancer_progress, 'axons')

        os.makedirs(self.progress_dir, exist_ok=True)
        os.makedirs(self.output, exist_ok=True)


        print(f'Input: {self.input}')
        print(f'Output: {self.output}')
        print(f'Progress: {self.progress_dir}')
        print(f'Rechunkme: {self.rechunkme_path}')
        if os.path.exists(self.rechunkme_path):
            print(f'Rechunkme path already exists: {self.rechunkme_path}')

        if is_non_empty_dir(self.output):
            print(f'Output path already exists and is not empty: {self.output}')
            isFinished = True

        return isFinished

    def create_label_density_map(self):
        """
        Create a cell density map based on the axon images.
        """
        print("Creating cell density map.")
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='axons')
        infiles = sorted([f for f in os.listdir(self.input) if f.endswith('.tif')])
        for infile in tqdm(infiles[20:22], desc="Creating cell density map", unit="image", disable=self.debug):
            input_path = os.path.join(self.input, infile)
            print(input_path)
            section_number = int(os.path.splitext(infile)[0])
            image = read_image(input_path)
            # Assuming the axon images are binary (0 and 65535), we can create a density map
            search_dict = {"FK_state_id": self.state_id, "layer_name": self.layer_name, "section": section_number}
            roi_row = self.sqlController.row_exists(search_dict, RegionOfInterest)

            self.create_contour_density_map(image, roi_row)


    def create_contour_density_map(self, img, roi_row: RegionOfInterest):
        ids = np.unique(img, return_counts=False)
        try:
            threshold = ids[1]  # Make sure this is between min and max above!
        except IndexError:
            return
        # Verify your actual pixel range so your threshold isn't too high
        print(f"Image min: {img.min()}, max: {img.max()} using {threshold=}")

        # 2. Threshold and convert to 8-bit binary
        _, binary_16 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        binary_8bit = binary_16.astype(np.uint8)

        # 3. Find all contours (OpenCV 4.x returns 2 values)
        contours, hierarchy = cv2.findContours(
            binary_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f'Found {len(contours)} contours')
        pixel_size_um = self.sqlController.scan_run.resolution  # microns per pixel
        pixel_area_um2 = pixel_size_um**2  # 0.105625 square microns per pixel
        min_area_px = 10.0

        shape_data = []
        for i, cnt in enumerate(contours):
            if cnt is not None and len(cnt) >= 3:
                area_px = cv2.contourArea(cnt)

                if area_px >= min_area_px:
                    # Calculate physical area in square microns
                    area_um2 = area_px * pixel_area_um2

                    # Calculate geometric center of mass (centroid) using image moments
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx_px = M["m10"] / M["m00"]
                        cy_px = M["m01"] / M["m00"]

                        # Convert center of mass coordinates to microns
                        cx_um = cx_px * pixel_size_um
                        cy_um = cy_px * pixel_size_um

                        shape_data.append({
                            "contour": cnt,
                            "area_px": area_px,
                            "area_um2": area_um2,
                            "centroid_px": (cx_px, cy_px),
                            "centroid_um": (cx_um, cy_um),
                        })

        # 5. Sort shapes by area from largest to smallest
        sorted_shapes = sorted(shape_data, key=lambda x: x["area_px"], reverse=True)
        update_dict = {"contour_count": len(sorted_shapes)}
        self.sqlController.update_row(RegionOfInterest, roi_row.id, update_dict)
        # should we delete the data below
        #self.sqlController.delete_rows_by_foreign_key(CellData, "FK_roi_id", roi_row.id)

        for shape in sorted_shapes:
            cx_px, cy_px = shape["centroid_px"]
            cx_um, cy_um = shape["centroid_um"]
            area_um2 = shape["area_um2"]
            area_px = shape["area_px"]
            print(area_px, area_um2)

            FK_roi_id = roi_row.id
            cell_data = CellData(FK_roi_id=FK_roi_id, com_x=cx_um, com_y=cy_um, area=area_um2)
            self.sqlController.add_row(cell_data)




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


