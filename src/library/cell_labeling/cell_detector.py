#ORG FILE BELOW; CONSOLIDATE


import cv2
import pandas as pd
import numpy as np
import os
from numpy.linalg import norm
from time import time
import glob

#from cell_extractor.CellDetectorIO import CellDetectorIO, parallel_process_all_sections


class ExampleFinder(CellDetectorIO):
    """class for finding examples (cell candidates) from image tiles

    :param CellDetectorIO: _description_
    :type CellDetectorIO: _type_
    """

    def __init__(self, animal, section, *args, **kwargs):
        super().__init__(animal, section, *args, **kwargs)
        self.t0 = time()
        print('section=%d, SECTION_DIR=%s, threshold=%d' % (
        self.section, self.fluorescence_channel_output_SECTION_DIR, self.segmentation_threshold))
        self.radius = 40
        self.max_segment_size = 100000
        self.cell_counter = 0
        self.Examples = []

    def find_examples(self):
        """finding example for one tile
        """
        if not os.path.exists(self.get_example_save_path()) or self.replace:
            self.load_manual_annotation()
            for tile in range(10):
                print(f'finding example for tile {tile}')
                self.load_and_preprocess_image(tile)
                self.find_connected_segments(self.difference_ch3)
                if self.n_segments > 2:
                    self.find_segments_corresponding_to_manual_labels(tile)
                    tilei_examples = self.get_examples(tile)
                    self.Examples.append(tilei_examples)

    def find_segments_corresponding_to_manual_labels(self, tile):
        """assign manual labels to tile (depricated)

        :param tile: _description_
        :type tile: _type_
        """
        self.load_manual_labels_in_tilei(tile)
        self.is_possitive_segment = np.zeros(self.n_segments)
        if self.n_manual_label > 0:
            for labeli in range(self.n_manual_label):
                manual_label = self.manual_labels_in_tile[labeli]
                cloest_segment_id = self.find_cloest_connected_segment_to_manual_label(manual_label)
                self.is_possitive_segment[cloest_segment_id] = 1

    def load_manual_annotation(self):
        """loads manual annotation (depricated)
        """
        file_list = glob.glob(
            os.path.join(self.fluorescence_channel_output_SECTION_DIR, f'*premotor*{self.section}*.csv'))
        if file_list != []:
            dfpath = file_list[0]
            self.manual_annotation = pd.read_csv(dfpath)
        else:
            self.manual_annotation = None

    def get_examples(self, tile):
        """creates examples for one tile

        :param tile: tile id
        :type tile: _type_
        :return: examples
        :rtype: _type_
        """
        origin = self.get_tile_origin(tile)
        Examples = []
        for segmenti in range(self.n_segments):
            _, _, width, height, area = self.segment_stats[segmenti, :]
            if area > self.max_segment_size:
                continue
            segment_row, segment_col = self.segment_location[segmenti, :]
            row_start = int(segment_row - self.radius)
            col_start = int(segment_col - self.radius)
            if row_start < 0 or col_start < 0:
                continue
            row_end = int(segment_row + self.radius)
            col_end = int(segment_col + self.radius)
            if row_end > self.tile_height or col_end > self.tile_width:
                continue
            segment_mask = self.segment_masks[row_start:row_end, col_start:col_end] == segmenti
            example = {'animal': self.animal,
                       'section': self.section,
                       'index': self.cell_counter,
                       'label': int(self.is_possitive_segment[segmenti]), #comment out; no labels
                       'area': area,
                       'row': segment_row,
                       'col': segment_col,
                       'origin': origin,
                       'height': height,
                       'width': width,
                       'image_CH3': self.difference_ch3[row_start:row_end, col_start:col_end],
                       'image_CH1': self.difference_ch1[row_start:row_end, col_start:col_end],
                       'mask': segment_mask}
            self.cell_counter += 1
            Examples.append(example)
        return Examples

    def get_tilei(self, tilei, channel='fluorescence'):
        """load image of tilei"""
        folder = getattr(self, f'{channel}_channel_output_SECTION_DIR')
        file = f'{self.section:03}tile-{tilei}.tif'
        infile = os.path.join(folder, file)
        img = np.float32(cv2.imread(infile, -1))
        return img

    def subtract_blurred_image(self, image):
        """average the image by subtracting gaussian blurred mean"""
        small = cv2.resize(image, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, ksize=(21, 21), sigmaX=10)
        relarge = cv2.resize(blurred, image.T.shape, interpolation=cv2.INTER_AREA)
        difference = image - relarge
        return difference

    def load_manual_labels_in_tilei(self, tilei):
        """load all manual labels that lies in the tile (depricated)"""
        if type(self.manual_annotation) != type(None):
            manual_annotation_array = self.manual_annotation[['y', 'x']]
            self.manual_labels_in_tile, self.n_manual_label = \
                self.get_manual_annotation_in_tilei(manual_annotation_array, tilei)
        else:
            self.n_manual_label = 0

    def find_connected_segments(self, image):
        """find connected segments (cell candidates)"""
        self.n_segments, self.segment_masks, self.segment_stats, self.segment_location \
            = cv2.connectedComponentsWithStats(np.int8(image > self.segmentation_threshold))
        self.segment_location = np.int32(self.segment_location)
        self.segment_location = np.flip(self.segment_location, 1)

    def load_and_preprocess_image(self, tile):
        """load and preprocess image for tilei"""
        self.fluorescence_channel_output_image = self.get_tilei(tile, channel='fluorescence')
        self.cell_body_channel_output_image = self.get_tilei(tile, channel='cell_body')
        self.difference_ch3 = self.subtract_blurred_image(self.fluorescence_channel_output_image)
        self.difference_ch1 = self.subtract_blurred_image(self.cell_body_channel_output_image)

    def find_cloest_connected_segment_to_manual_label(self, manual_label):
        """pairing manual label to automatic detection (depricated)"""
        self.segment_distance_to_label = norm(self.segment_location - manual_label, axis=1)
        self.cloest_segment_id = np.argmin(self.segment_distance_to_label)
        return self.cloest_segment_id


def create_examples_for_all_sections(animal, *args, njobs=10, **kwargs):
    """extract examples for all sections in parallel"""
    parallel_process_all_sections(animal, create_examples_for_one_section, *args, njobs=njobs, **kwargs)


def create_examples_for_one_section(animal, section, *args, **kwargs):
    """extract example for one section"""
    extractor = ExampleFinder(animal=animal, section=section, *args, **kwargs)
    extractor.find_examples()
    extractor.save_examples()
