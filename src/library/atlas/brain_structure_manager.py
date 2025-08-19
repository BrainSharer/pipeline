import ast
import math
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from collections import defaultdict
import cv2
import json
import pandas as pd
from scipy.ndimage import center_of_mass, zoom
from skimage.filters import gaussian

from cloudvolume import CloudVolume
import sqlalchemy
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import SimpleITK as sitk

from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from tqdm import tqdm


from library.image_manipulation.image_manager import ImageManager
from library.image_manipulation.pipeline_process import Pipeline
from library.atlas.atlas_manager import AtlasToNeuroglancer
from library.atlas.atlas_utilities import (
    adjust_volume,
    affine_transform_point,
    compute_affine_transformation,
    get_evenly_spaced_vertices,
    get_evenly_spaced_vertices_from_slice,
    get_min_max_mean,
    list_coms,
    ORIGINAL_ATLAS,
    load_transformation,
    order_points_concave_hull,
    interpolate_points
)
from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationLabel, AnnotationSession
from library.image_manipulation.filelocation_manager import (
    data_path,
    FileLocationManager,
)
from library.utilities.atlas import volume_to_polygon, save_mesh
from library.utilities.utilities_process import (
    M_UM_SCALE,
    SCALING_FACTOR,
    get_hostname,
    get_image_size,
    random_string,
    read_image,
    write_image,
)
from library.utilities.utilities_contour import get_contours_from_annotations


class BrainStructureManager:

    def __init__(self, animal, um=10, affine=False, scaling_factor=SCALING_FACTOR, debug=False):

        self.animal = animal
        self.fixed_brain = None
        self.sqlController = SqlController(animal)
        self.fileLocationManager = FileLocationManager(self.animal)
        self.data_path = os.path.join(data_path, "atlas_data")
        self.structure_path = os.path.join(data_path, "pipeline_data", "structures")
        self.com_path = os.path.join(self.data_path, self.animal, "com") 
        self.registered_com_path = os.path.join(self.data_path, self.animal, "registered_com")
        self.origin_path = os.path.join(self.data_path, self.animal, "origin")
        self.mesh_path = os.path.join(self.data_path, self.animal, "mesh")
        self.volume_path = os.path.join(self.data_path, self.animal, "structure")

        self.debug = debug
        self.um = um  # size in um of allen atlas
        self.com = None
        self.origin = None
        self.volume = None
        self.abbreviation = None
        self.pad_z = 0 # set to 40 to make nice rounded edges on the big structures

        self.affine = affine
        self.scaling_factor = scaling_factor
        self.atlas_box_scales = np.array((self.um, self.um, self.um))

        self.allen_x_length = 1820 + 0
        self.allen_y_length = 1000 + 0
        self.allen_z_length = 1140 + 0
        #unpadded_allen_x_length = 1320
        #unpadded_allen_y_length = 800
        #unpadded_allen_z_length = 1140
        self.atlas_box_size = np.array(
            (self.allen_x_length, self.allen_y_length, self.allen_z_length)
        )
        self.atlas_box_center = self.atlas_box_size / 2

        os.makedirs(self.com_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)


    def get_origin_and_section_size(self, structure_contours):
        """Gets the origin and section size
        Set the pad to make sure we get all the volume
        """
        section_mins = []
        section_maxs = []
        for _, contour_points in structure_contours.items():
            contour_points = np.array(contour_points)
            section_mins.append(np.min(contour_points, axis=0))
            section_maxs.append(np.max(contour_points, axis=0))
        min_z = min([int(i) for i in structure_contours.keys()])
        min_x, min_y = np.min(section_mins, axis=0)
        max_x, max_y = np.max(section_maxs, axis=0)

        xspan = max_x - min_x
        yspan = max_y - min_y
        origin = np.array([min_x, min_y, min_z])
        # flipped yspan and xspan 19 Oct 2023
        section_size = np.array([yspan, xspan]).astype(int)
        return origin, section_size

    def create_foundation_brains_origin_volume(self, brainMerger, animal):
        """ """
        self.animal = animal
        com_path = os.path.join(self.data_path, self.animal, "com")
        origin_path = os.path.join(self.data_path, self.animal, "origin")
        volume_path = os.path.join(self.data_path, self.animal, "structure")
        self.check_for_existing_dir(com_path)
        self.check_for_existing_dir(origin_path)
        self.check_for_existing_dir(volume_path)
        coms = sorted(os.listdir(com_path))
        origins = sorted(os.listdir(origin_path))
        volumes = sorted(os.listdir(volume_path))

        # loop through structure objects
        for com_file, origin_file, volume_file in zip(coms, origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(f"{Path(origin_file).stem} and {Path(volume_file).stem} do not match")
                sys.exit()
            structure = Path(origin_file).stem
            com = np.loadtxt(os.path.join(com_path, com_file))
            origin = np.loadtxt(os.path.join(origin_path, origin_file))
            volume = np.load(os.path.join(volume_path, volume_file))

            if self.debug:
                print(f"{animal} {structure} origin={np.round(origin)} com={np.round(com)}")
            else:
                # merge data
                brainMerger.coms_to_merge[structure].append(com)
                brainMerger.origins_to_merge[structure].append(origin)
                brainMerger.volumes_to_merge[structure].append(volume)

    def get_label_ids(self, structure) -> list:
        label = self.sqlController.get_annotation_label(structure)
        if label is not None:
            label_ids = [label.id]
        else:
            print(f"Could not find {structure} label in database")
            label_ids = [0]

        return label_ids

    def create_brains_origin_volume_from_polygons(self, brainMerger, animal, structure, debug=False):

        label_ids = self.get_label_ids(structure)
        annotator_id = 1 # hard coded to Edward

        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.FK_prep_id == animal)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .one_or_none()
            )
        except Exception as e:
            print(f"Error occurred while fetching annotation session for {animal}: {structure}")
            print(e)
            return

        if annotation_session is None:
            print(f'Did not find any data for {animal} {structure}')
            return
        
        # Inverse = True as we are transforming points.
        transform = load_transformation(animal, self.um, self.um, inverse=False)
        if transform is None:
            print(f'Could not load transform for {animal}')
            return

        polygons = self.sqlController.get_annotation_volume(annotation_session.id, self.um)
        if len(polygons) == 0:
            print(f'Found data for {animal} {structure}, but the data is empty')
            return


        origin, volume = self.create_volume_for_one_structure_from_polygons(polygons, self.pad_z, transform)
        # we want to keep the origin in micrometers, so we multiply by the allen um
        #####origin = origin * self.um

        if origin is None or volume is None:
            print(f"{animal} {structure} has no volumes to merge")
            return None
        
        volume = np.swapaxes(volume, 0, 2)
        #volume = gaussian(volume, 1.0)
        #volume[volume != 0] = 255 # set all values that are not zero to 255, which is the drawn shape value
        #volume = volume.astype(np.uint8)
        #com = (np.array( center_of_mass(volume) ) - self.pad_z) * self.um + (origin * self.um)
        com = (np.array( center_of_mass(volume) ))  + origin
        if debug:
            print(f'ID={annotation_session.id} animal={animal} {structure} origin={np.round(origin)} com={np.round(com)} polygon len {len(polygons)}')
        else:
            brainMerger.coms_to_merge[structure].append(com)
            brainMerger.origins_to_merge[structure].append(origin)
            brainMerger.volumes_to_merge[structure].append(volume)

            structure_path = os.path.join(self.data_path, animal, "structure2allen")
            os.makedirs(structure_path, exist_ok=True)
            np.save(os.path.join(structure_path, f"{structure}.npy"), volume)


    def save_brain_origins_and_volumes_and_meshes(self):
        """Saves everything to disk, no calculations, only saving!"""

        aligned_structure = volume_to_polygon(
            volume=self.volume, origin=self.origin, times_to_simplify=3
        )

        com_filepath = os.path.join(self.com_path, f"{self.abbreviation}.txt")
        origin_filepath = os.path.join(self.origin_path, f"{self.abbreviation}.txt")
        mesh_filepath = os.path.join(self.mesh_path, f"{self.abbreviation}.stl")
        volume_filepath = os.path.join(self.volume_path, f"{self.abbreviation}.npy")

        np.savetxt(com_filepath, self.com)
        np.savetxt(origin_filepath, self.origin)
        save_mesh(aligned_structure, mesh_filepath)
        np.save(volume_filepath, self.volume)

    def get_allen_id(self, structure: str) -> int:

        label = self.sqlController.get_annotation_label(structure)
        if label is None:
            print(f"Could not find {structure} label in database")
            sys.exit()
        else:
            allen_id = label.allen_id

        if allen_id is None:
            print(f"Could not find {structure} allen_id in database")
            print(
                f"Please update the database with an ID in the allen_id column for {structure}"
            )
            sys.exit()

        return allen_id

    def list_coms_by_atlas(self):
        structures = list_coms(self.animal)
        xy_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        for structure, com in structures.items():
            # com = scale_coordinate(com, self.animal)
            print(f"{structure}={com}")

    def update_database_com(self, animal: str, structure: str, com: np.ndarray, um=1) -> None:
        """Annotator ID is hardcoded to 2 for Beth
        Data coming in is in pixels, so we need to convert to um and then to meters
        """
        annotator_id = 2

        com = com.tolist()
        x = com[0] * um / M_UM_SCALE
        y = com[1] * um / M_UM_SCALE
        z = com[2] * um / M_UM_SCALE
        com = [x, y, z]
        json_entry = {
            "type": "point",
            "point": com,
            "description": f"{structure}",
            "centroid": com,
            "props": ["#ffff00", 1, 1, 5, 3, 1],
        }
        label = self.sqlController.get_annotation_label(structure)
        if label is not None:
            label_ids = [label.id]
        else:
            print(f"Could not find {structure} label in database")
            return
        # update label with allen ID
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == animal)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .filter(AnnotationSession.annotation["type"] == "point")
                .one()
            )
        except NoResultFound as nrf:
            print(f"Inserting {structure} with {com}")
            self.sqlController.insert_annotation_with_labels(
                FK_user_id=annotator_id,
                FK_prep_id=animal,
                annotation=json_entry,
                labels=[structure],
            )
            return

        except MultipleResultsFound as mrf:
            print(f"Multiple results found for {structure} error: {mrf}")
            return

        if annotation_session:
            print(f"Updating {structure} with {com} with ID={annotation_session.id}")
            update_dict = {"annotation": json_entry}
            self.sqlController.update_session(
                annotation_session.id, update_dict=update_dict
            )

    @staticmethod
    def check_for_existing_dir(path):
        if not os.path.exists(path):
            print(f"{path} does not exist, exiting.")
            sys.exit()

    @staticmethod
    def rm_existing_dir(path):
        if os.path.exists(path):
            print(f"{path} exists, removing ...")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def get_transformed(self, point):
        new_point = affine_transform_point(point)
        return new_point
    
    @staticmethod
    def get_start_positions(volume: np.ndarray, com: tuple) -> tuple:
        COM = center_of_mass(volume) 
        x_start = int(com[0] - COM[0])
        y_start = int(com[1] - COM[1])
        z_start = int(com[2] - COM[2])
        return x_start, y_start, z_start


    def get_transformation_matrix(self):
        moving_name = 'AtlasV8'
        fixed_name = 'Allen'
        moving_all = list_coms(moving_name, scaling_factor=self.um)
        fixed_all = list_coms(fixed_name, scaling_factor=self.um)
        bad_keys = ('RtTg', 'AP')
        common_keys = list(moving_all.keys() & fixed_all.keys())
        good_keys = set(common_keys) - set(bad_keys)
        moving_src = np.array([moving_all[s] for s in good_keys])
        fixed_src = np.array([fixed_all[s] for s in good_keys])
        return compute_affine_transformation(moving_src, fixed_src)

    def create_atlas_volume(self):
        self.check_for_existing_dir(self.com_path)
        self.check_for_existing_dir(self.volume_path)

        ### test using aligned images from foundation brain
        if 'MD' in self.animal:
            xy_resolution = self.sqlController.scan_run.resolution * SCALING_FACTOR /  self.um
            z_resolution = self.sqlController.scan_run.zresolution / self.um
            input_path = os.path.join(self.fileLocationManager.prep, "C1", "thumbnail_aligned")
            image_manager = ImageManager(input_path)
            self.atlas_box_size = image_manager.volume_size
            self.atlas_box_size = np.array([int(self.atlas_box_size[0] * xy_resolution), 
                                               int(self.atlas_box_size[1] * xy_resolution),
                                               int(self.atlas_box_size[2] * z_resolution)])
        atlas_volume = np.zeros((self.atlas_box_size), dtype=np.uint32)
        print(f"atlas box size={self.atlas_box_size} shape={atlas_volume.shape}")
        print(f"Using data from {self.com_path}")
        coms = sorted(os.listdir(self.com_path)) # registered COMs are in micrometers/self.um
        origins = sorted(os.listdir(self.origin_path)) # origins are in micrometers/self.um
        volumes = sorted(os.listdir(self.volume_path))
        if len(coms) != len(volumes):
            print(f'The number of coms: {len(coms)} does not match the number of volumes: {len(volumes)}')
            sys.exit()

        print(f"Working with {len(coms)} coms/volumes from {self.com_path}")
        ids = {}
        if self.affine:
            transformation_matrix = self.get_transformation_matrix()

        for com_file, origin_file, volume_file in zip(coms, origins, volumes):
            if Path(com_file).stem != Path(volume_file).stem:
                print(f"{Path(com_file).stem} and {Path(volume_file).stem} do not match")
                sys.exit()
            structure = Path(com_file).stem
            allen_id = self.get_allen_id(structure)
            try:
                ids[structure] = allen_id
            except IndexError as ke:
                print(f"Problem with index error: {structure=} {allen_id=} in database")
                sys.exit()

            com0 = np.loadtxt(os.path.join(self.com_path, com_file))
            # origin0 is already in 10um space
            origin0 = np.loadtxt(os.path.join(self.origin_path, origin_file))
            # com0 is in micrometers, so convert to allen space
            com0 = com0 / self.um

            volume0 = np.load(os.path.join(self.volume_path, volume_file))
            if self.animal == ORIGINAL_ATLAS:
                volume = np.rot90(volume, axes=(0, 1))
                volume = np.flip(volume, axis=0)

            volume = adjust_volume(volume0, allen_id)
            #x_start, y_start, z_start = self.get_start_positions(volume, com0)

            if self.affine:
                com = affine_transform_point(com0, transformation_matrix)
                origin = affine_transform_point(origin0, transformation_matrix)
                #x_start, y_start, z_start = affine_transform_point((x_start, y_start, z_start), transformation_matrix)
            else:
                com = com0
                origin = origin0

            if structure in ['TG_L', 'TG_R']:
                #origin = [1048, 157, 191]
                #origin =(1006.4404940906367, 66.58640497850439, 201.97515475575017)
                volume = volume0.astype(np.uint32)
                upper = 100
                volume[(volume > 0)] = allen_id
                #volume[(volume != allen_id)] = 0
                #volume = volume.astype(np.uint32)

                com = com0
                origin = origin0
                print(f"Using {structure} origin {origin} com {com}")

            #x_start, y_start, z_start = self.get_start_positions(volume, com)

            # Using the origin makes the structures appear a bit too far up
            x_start = int(round(origin[0]))
            y_start = int(round(origin[1]))
            z_start = int(round(origin[2]))

            x_end = x_start + volume.shape[0]
            y_end = y_start + volume.shape[1]
            z_end = z_start + volume.shape[2]

            if self.debug:
                print(f"{structure} com={np.round(com)}", end = " ")
                nids, ncounts = np.unique(volume, return_counts=True)
                print(f"x={x_start}:{x_end} y={y_start}:{y_end} z={z_start}:{z_end} ids={nids}, counts={ncounts} allen IDs={allen_id} dtype={volume.dtype}")
                if x_end > atlas_volume.shape[0]:
                    print(f"\tWarning: End x {x_end} is larger than atlas volume shape {atlas_volume.shape[0]}")
                if y_end > atlas_volume.shape[1]:
                    print(f"\tWarning: End y {y_end} is larger than atlas volume shape {atlas_volume.shape[1]}")
                if z_end > atlas_volume.shape[2]:
                    print(f"\tWarning: End z {z_end} is larger than atlas volume shape {atlas_volume.shape[2]}")
            else:
                try:
                    atlas_volume[x_start:x_end, y_start:y_end, z_start:z_end] += volume
                except ValueError as ve:
                    print(f"Error adding {structure} to atlas: {ve}")
                    print(f"{structure} com={np.round(com)}", end = " ") 
                    print(f"x={x_start}:{x_end} y={y_start}:{y_end} z={z_start}:{z_end}")
                    #sys.exit()


        if self.affine:        
            print(f"Transformation matrix\n {transformation_matrix}")

        print(f"Atlas volume shape={atlas_volume.shape} dtype={atlas_volume.dtype}")
        return atlas_volume, ids

    def update_atlas_coms(self) -> None:
        """
        Updates the center of mass (COM) for each structure in the atlas.

        This method creates an atlas volume and retrieves the center of mass for each structure.
        If the debug mode is enabled, it prints the center of mass for each structure.
        Otherwise, it updates the database with the new center of mass values.
        If you are updating the COMS, they need to be calculated from the correct
        box size. The box size is defined in the `self.atlas_box_size` attribute.

        Returns:
            None
        """
        if self.animal is None:
            brains = ['MD585', 'MD589', 'MD594', 'AtlasV8']
        else:
            brains = [self.animal]

        if self.debug:

            for animal in brains:
                com_path = os.path.join(self.data_path, animal, "com")
                files = sorted(os.listdir(com_path))
                for file in files:
                    comfile_path = os.path.join(com_path, file)
                    com = np.loadtxt(comfile_path)
                    structure = file.replace(".txt", "")
                    print(f"{animal} {structure}={np.round(com)}")
        else:
            for animal in brains:
                com_path = os.path.join(self.data_path, animal, "com")
                files = sorted(os.listdir(com_path))
                for file in files:
                    comfile_path = os.path.join(com_path, file)
                    com = np.loadtxt(comfile_path)
                    structure = file.replace(".txt", "")
                    self.update_database_com(animal, structure, com, 1)


    def save_atlas_volume(self) -> None:
        """
        Saves the atlas volume to a specified file path.

        This method creates an atlas volume and saves it to a predefined location.
        If the file already exists at the specified location, it will be removed
        before saving the new atlas volume.

        Returns:
            None
        """

        atlas_volume, ids = self.create_atlas_volume()
        if not self.debug:
            outpath = os.path.join(self.data_path, self.animal, f"DK{self.animal}_{self.um}um_sagittal.tif")
            midpath = os.path.join(self.data_path, self.animal, f"DK{self.animal}_{self.um}um_midpath.tif")
            jsonpath = os.path.join(self.data_path, self.animal, "ids.json")

            if os.path.exists(outpath):
                print(f"Removing {outpath}")
                os.remove(outpath)
            print(f"saving image to {outpath}")
            write_image(outpath, atlas_volume)

            if os.path.exists(midpath):
                print(f"Removing {midpath}")
                os.remove(outpath)
            print(f"saving image to {midpath}")
            midpoint = int(atlas_volume.shape[2] / 4)
            midvolume = atlas_volume[:, :, midpoint].astype(np.uint16)
            write_image(midpath, midvolume)
            print(f"Saving ids to {jsonpath}")
            with open(jsonpath, "w") as f:
                json.dump(ids, f, indent=4)

    def create_neuroglancer_volume(self):
        """
        Creates a Neuroglancer volume from the atlas volume.
        Note, COMs are saved in micrometers, but the volumes and origins
        get saved in 10um allen space.

        This method performs the following steps:
        1. Creates the atlas volume by calling `self.create_atlas_volume()`.
        2. Prints the shape and data type of the atlas volume before and after processing.
        3. If not in debug mode:
            a. Removes the existing structure directory if it exists.
            b. Creates a new structure directory.
            c. Initializes a Neuroglancer volume with the atlas volume and scales.
            d. Adds segment properties to the Neuroglancer volume.
            e. Adds downsampled volumes to the Neuroglancer volume.
            f. Adds segmentation mesh to the Neuroglancer volume.


        Raises:
            OSError: If there is an issue creating or removing directories.

        """
        # check if atlas volume and json already exist
        outpath = os.path.join(self.data_path, self.animal, f"DK{self.animal}_{self.um}um_sagittal.tif")
        jsonpath = os.path.join(self.data_path, self.animal, "ids.json")
        print(f"Checking for {outpath}")
        print(f"Checking for {jsonpath}")
        if os.path.exists(outpath) and os.path.exists(jsonpath):
            print(f"Atlas volume and json already exist")
            #atlas_volume = read_image(outpath)
            #with open(jsonpath, "r") as f:
            #    ids = json.load(f)
        else:
            print(f"Atlas volume or json do not exist, creating new ...")

        ##### Actual atlas volume is in this step
        atlas_volume, ids = self.create_atlas_volume()

        print(f"Pre Shape {atlas_volume.shape=} {atlas_volume.dtype=}")

        if not self.debug:
            aligned = "aligned" if self.affine else "unaligned"
            atlas_name = f"DK.{self.animal}.{aligned}.{self.um}um"
            structure_path = os.path.join(self.structure_path, atlas_name)
            if os.path.exists(structure_path):
                if 'tobor' in get_hostname() or 'mothra' in get_hostname():
                    print(f"Removing {structure_path}")
                    shutil.rmtree(structure_path)
                else:
                   print(f"Path exists, please remove first: {structure_path}")
                   sys.exit(1)
            else:
                print(f"Creating data in {structure_path}")
            os.makedirs(structure_path, exist_ok=True)
            self.atlas_box_scales = np.array(
                (int(self.atlas_box_scales[0] * 1000), 
                 int(self.atlas_box_scales[1] * 1000), 
                 int(self.atlas_box_scales[2] * 1000))
            )
            neuroglancer = AtlasToNeuroglancer(volume=atlas_volume, scales=self.atlas_box_scales)
            neuroglancer.init_precomputed(path=structure_path)
            neuroglancer.add_segment_properties(ids)
            neuroglancer.add_downsampled_volumes()
            neuroglancer.add_segmentation_mesh()

    def create_foundation_brain_volumes_origins(self, brainMerger, animal, debug):
        """Step 2
        We want to save the COMs in micrometers, but the volumes and origins
        get saved in 10um allen space
        """

        jsonpath = os.path.join(
            self.data_path, animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            sys.exit()
        with open(jsonpath) as f:
            aligned_dict = json.load(f)

        xy_resolution = self.fixed_brain.sqlController.scan_run.resolution
        zresolution = self.fixed_brain.sqlController.scan_run.zresolution
        
        structures = list(aligned_dict.keys())
        desc = f"Create {animal} coms/meshes/origins/volumes"
        for structure in tqdm(structures, desc=desc, disable=debug):
            polygons = aligned_dict[structure]
            origin0, volume = self.create_volume_for_one_structure(polygons, self.pad_z)
            pads = np.array([0, 0, self.pad_z])
            com0 = center_of_mass(np.swapaxes(volume, 0, 2)) - pads + origin0

            # Now convert com and origin to micrometers
            scale0 = np.array([xy_resolution*SCALING_FACTOR, xy_resolution*SCALING_FACTOR, zresolution])
            com_um = com0 * scale0 # COM in um
            origin_um = origin0 * scale0 # origin in um

            # we want the volume and origin scaled to 10um, so adjust the above scale
            scale_allen = scale0 / self.um
            origin_allen = origin_um / self.um
            volume = np.swapaxes(volume, 0, 2) # put into x,y,z order
            volume_allen = zoom(volume, scale_allen)

            if debug:
                if structure == 'SC':
                    print(f"ID={animal} {structure} com={np.round(com_um)}um", end=" ")
                    print(f"origin0={np.round(origin0)} origin um={np.round(origin_um)}um \
                        origin allen(10um)={np.round(origin_allen)} com allen{np.round(com_um/self.um)}")
            else:
                brainMerger.coms[structure] = com_um
                brainMerger.origins[structure] = origin_allen
                brainMerger.volumes[structure] = volume_allen

    @staticmethod
    def save_volume_origin(animal, structure, volume, xyz_offsets):
        x, y, z = xyz_offsets

        volume = np.swapaxes(volume, 0, 2)
        volume = np.rot90(volume, axes=(0, 1))
        volume = np.flip(volume, axis=0)

        OUTPUT_DIR = os.path.join(data_path, "atlas_data", animal)
        volume_filepath = os.path.join(OUTPUT_DIR, "structure", f"{structure}.npy")
        print(f"Saving {animal=} {structure=} to {volume_filepath}")
        os.makedirs(os.path.join(OUTPUT_DIR, "structure"), exist_ok=True)
        np.save(volume_filepath, volume)
        origin_filepath = os.path.join(OUTPUT_DIR, "origin", f"{structure}.txt")
        os.makedirs(os.path.join(OUTPUT_DIR, "origin"), exist_ok=True)
        np.savetxt(origin_filepath, (x, y, z))

    def test_brain_volumes_and_origins(self, animal):
        jsonpath = os.path.join(
            self.data_path, animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            sys.exit()
        with open(jsonpath) as f:
            aligned_dict = json.load(f)
        structures = list(aligned_dict.keys())
        input_directory = os.path.join(
            self.fileLocationManager.prep, "C1", "thumbnail_aligned"
        )
        files = sorted(os.listdir(input_directory))
        print(f"Working with {len(files)} files")
        if not os.path.exists(input_directory):
            print(f"{input_directory} does not exist")
            sys.exit()
        drawn_directory = os.path.join(self.fileLocationManager.prep, "C1", "drawn")
        if os.path.exists(drawn_directory):
            print(f"Removing {drawn_directory}")
            shutil.rmtree(drawn_directory)
        os.makedirs(drawn_directory, exist_ok=True)

        desc = f"Drawing on {animal}"
        for tif in tqdm(files, desc=desc):
            infile = os.path.join(input_directory, tif)
            outfile = os.path.join(drawn_directory, tif)
            file_section = int(tif.split(".")[0])
            img = read_image(infile)
            for structure in structures:
                onestructure = aligned_dict[structure]
                for section, points in sorted(onestructure.items()):
                    if int(file_section) == int(section):
                        vertices = np.array(points)
                        points = (vertices).astype(np.int32)
                        cv2.polylines(
                            img, [points], isClosed=True, color=255, thickness=2
                        )

            write_image(outfile, img)

    def create_cloud_volume(self):
        jsonpath = os.path.join(
            self.data_path, self.animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            sys.exit()
        with open(jsonpath) as f:
            aligned_dict = json.load(f)
        structures = list(aligned_dict.keys())
        input_directory = os.path.join(
            self.fileLocationManager.prep, "C1", "thumbnail_aligned"
        )
        if not os.path.exists(input_directory):
            print(f"{input_directory} does not exist")
            sys.exit()
        xyresolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution

        w = int(self.sqlController.scan_run.width//SCALING_FACTOR)
        h = int(self.sqlController.scan_run.height//SCALING_FACTOR)
        z_length = len(os.listdir(input_directory))
        shape = (w, h, z_length)

        drawn_directory = os.path.join(self.fileLocationManager.neuroglancer_data, 'polygons')
        if os.path.exists(drawn_directory):
            print(f"Removing {drawn_directory}")
            shutil.rmtree(drawn_directory)
        os.makedirs(drawn_directory, exist_ok=True)
        volume = np.zeros((z_length, h, w), dtype=np.uint8)  # Initialize the volume with zeros
        desc = f"Drawing on {self.animal} volume={volume.shape}"
        for z in tqdm(range(volume.shape[0]), desc=desc):
            volume_slice = np.zeros((h, w), dtype=np.uint8)  # Create a slice for the current z
            for structure in structures:
                onestructure = aligned_dict[structure]
                for section, points in sorted(onestructure.items()):
                    if int(z) == int(section):
                        vertices = np.array(points)
                        points = (vertices).astype(np.int32)
                        cv2.polylines(volume_slice, [points], isClosed=True, color=255, thickness=2)
                        #cv2.fillPoly(volume_slice, [points], color=200)
            volume[z,:,:] = volume_slice

        resolution = int(xyresolution * 1000 * SCALING_FACTOR)
        #ids, counts = np.unique(volume, return_counts=True)
        #print(f"Unique ids {ids} counts {counts}")
        volume = np.swapaxes(volume, 0, 2)          
        #return
        scales = (resolution, resolution, int(zresolution * 1000))
        
        
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='image',  # or 'segmentation' if you're using labels
            data_type=np.uint8,   # or 'uint32' for segmentation
            encoding='raw',
            resolution=scales,
            voxel_offset=(0, 0, 0),
            chunk_size=(32,32,32),
            #volume_size=volume.shape[::-1],  # x,y,z
            volume_size=shape,  # x,y,z
        )
        tq = LocalTaskQueue(parallel=1)

        vol = CloudVolume(f'file://{drawn_directory}', info=info)
        vol.commit_info()
        vol[:,:,:] = volume
        tasks = tc.create_downsampling_tasks(f'file://{drawn_directory}', mip=0, num_mips=2, compress=True)
        tq.insert(tasks)
        tq.execute()


    ### Imported methods from old build_foundationbrain_aligned data
    def create_clean_transform(self, animal):
        sqlController = SqlController(animal)
        fileLocationManager = FileLocationManager(animal)
        aligned_shape = np.array(
            (sqlController.scan_run.width, sqlController.scan_run.height)
        )
        # downsampled_aligned_shape = np.round(aligned_shape / DOWNSAMPLE_FACTOR).astype(int)
        downsampled_aligned_shape = aligned_shape / 32
        print(f"downsampled shape {downsampled_aligned_shape}")
        INPUT = os.path.join(fileLocationManager.prep, "C1", "thumbnail")
        files = sorted(os.listdir(INPUT))
        section_offsets = {}
        for file in tqdm(files):
            filepath = os.path.join(INPUT, file)
            width, height = get_image_size(filepath)
            # width = int(width)
            # height = int(height
            downsampled_shape = np.array((width, height))
            section = int(file.split(".")[0])
            section_offsets[section] = (
                downsampled_aligned_shape - downsampled_shape
            ) / 2
        return section_offsets

    def create_brain_json(self, animal, debug):

        pipeline = Pipeline(
            animal,
            channel=1,
            downsample=True,
            scaling_factor=SCALING_FACTOR,
            task=None,
            debug=debug,
        )

        section_offsets = self.create_clean_transform(animal)
        transforms = pipeline.get_transformations(iteration=0)

        # warp_transforms = create_downsampled_transforms(animal, transforms, downsample=True)
        ordered_downsampled_transforms = sorted(transforms.items())
        section_structure_vertices = defaultdict(dict)
        csvfile = os.path.join(
            self.data_path, "foundation_brain_annotations", f"{animal}_annotation.csv"
        )
        hand_annotations = pd.read_csv(csvfile)
        # below is the fix for section numbers. The original data started at 1, but we now start at 0
        hand_annotations["section"] = hand_annotations["section"] - 1
        hand_annotations["vertices"] = (
            hand_annotations["vertices"]
            .apply(lambda x: x.replace(" ", ","))
            .apply(lambda x: x.replace("\n", ","))
            .apply(lambda x: x.replace(",]", "]"))
            .apply(lambda x: x.replace(",,", ","))
            .apply(lambda x: x.replace(",,", ","))
            .apply(lambda x: x.replace(",,", ","))
            .apply(lambda x: x.replace(",,", ","))
        )

        hand_annotations["vertices"] = hand_annotations["vertices"].apply(
            lambda x: ast.literal_eval(x)
        )
        files = sorted(os.listdir(self.origin_path))
        structures = [str(structure).split(".")[0] for structure in files]

        for structure in structures:
            contour_annotations, first_sec, last_sec = get_contours_from_annotations(
                structure, hand_annotations, densify=0
            )
            for section in contour_annotations:
                section_structure_vertices[section][structure] = contour_annotations[
                    section
                ]

        section_transform = {}
        for section, transform in ordered_downsampled_transforms:
            section_num = int(section.split(".")[0])
            transform = np.linalg.inv(transform)
            section_transform[section_num] = transform

        md585_fixes = {160: 100, 181: 60, 222: 60, 228: 76, 230: 80, 252: 60}
        md589_fixes = {294: 0}
        original_structures = defaultdict(dict)
        unaligned_padded_structures = defaultdict(dict)
        aligned_padded_structures = defaultdict(dict)
        # Data in the vertices is in 0.452um/pixel and is downsampled by 1/32 to fit the smaller images
        for section in section_structure_vertices:
            section = int(section)
            for structure in section_structure_vertices[section]:

                points = np.array(section_structure_vertices[section][structure]) / SCALING_FACTOR
                original_structures[structure][section] = points.tolist()
                offset = section_offsets[section]
                if animal == "MD585" and section in md585_fixes.keys():
                    offset = offset - np.array([0, md585_fixes[section]])
                if animal == "MD589" and section == 294:
                    offset = offset + np.array([23, 8])
                if animal == "MD589" and section == 296:
                    offset = offset + np.array([8, 29])

                points = np.array(points) + offset
                unaligned_padded_structures[structure][section] = points.tolist()

                points = self.transform_create_alignment(
                    points, section_transform[section]
                )  # create_alignment transform
                aligned_padded_structures[structure][section] = points.tolist()

        if not debug:

            OUTPUT_DIR = os.path.join(self.data_path, animal)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"Saving data to {OUTPUT_DIR}")

            jsonpath1 = os.path.join(OUTPUT_DIR, "original_structures.json")
            with open(jsonpath1, "w") as f:
                json.dump(original_structures, f, sort_keys=True)

            jsonpath2 = os.path.join(OUTPUT_DIR, "unaligned_padded_structures.json")
            with open(jsonpath2, "w") as f:
                json.dump(unaligned_padded_structures, f, sort_keys=True)

            jsonpath3 = os.path.join(OUTPUT_DIR, "aligned_padded_structures.json")
            with open(jsonpath3, "w") as f:
                json.dump(aligned_padded_structures, f, sort_keys=True)

    def validate_volumes(self):
        self.check_for_existing_dir(self.origin_path)
        self.check_for_existing_dir(self.volume_path)

        origins = sorted(os.listdir(self.origin_path))
        volumes = sorted(os.listdir(self.volume_path))
        print(
            f"Testing {len(origins)} origins and {len(volumes)} volumes from {self.volume_path}."
        )
        errors = ""
        for origin_file, volume_file in zip(origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(
                    f"{Path(origin_file).stem} and {Path(volume_file).stem} do not match"
                )
                sys.exit()
            structure = Path(origin_file).stem
            volume = np.load(os.path.join(self.volume_path, volume_file))
            allen_id = 666  # we can use a fake number here.
            volume = adjust_volume(volume=volume, allen_id=allen_id)
            com = center_of_mass(volume)
            nids, ncounts = np.unique(volume, return_counts=True)
            if len(nids) == 1:
                errors += f"{structure} has only: IDs={nids} counts={ncounts} COM={com}, please check\n"
        if len(errors) > 0:
            print("There were errors:")
            print(errors)
        else:
            print("No errors found")

    def evaluate(self):
        def sum_square_com(com):
            ss = np.sqrt(sum([s * s for s in com]))
            return ss

        print(f"evaluating atlas data from {self.com_path}")

        atlas_all = list_coms(self.animal)
        allen_all = list_coms("Allen")
        bad_keys = ("RtTg", "AP")

        common_keys = sorted(list(atlas_all.keys() & allen_all.keys()))
        good_keys = set(common_keys) - set(bad_keys)

        atlas_src = np.array([atlas_all[s] for s in good_keys])
        allen_src = np.array([allen_all[s] for s in good_keys])
        matrix = compute_affine_transformation(atlas_src, allen_src)

        error = []
        for structure in common_keys:
            atlas0 = np.array(atlas_all[structure])
            allen0 = np.array(allen_all[structure])
            transformed = affine_transform_point(atlas0, matrix)
            transformed = [x for x in transformed]
            difference = [a - b for a, b in zip(transformed, allen0)]
            ss = sum_square_com(difference)
            error.append(ss)
            print(
                f"{structure} atlas={np.round(atlas0)} allen={np.round(allen0)} transformed={np.round( np.array(transformed)) } \
                error={np.round(np.array(difference))} ss={round(ss,2)}"
            )
        print("RMS", sum(error) / len(common_keys))


    def report_status(self) -> None:
        com_path = os.path.join(self.data_path, self.animal, "com")
        origin_path = os.path.join(self.data_path, self.animal, "origin")
        volume_path = os.path.join(self.data_path, self.animal, "structure")
        self.check_for_existing_dir(com_path)
        self.check_for_existing_dir(origin_path)
        self.check_for_existing_dir(volume_path)
        coms = sorted(os.listdir(com_path))
        origins = sorted(os.listdir(origin_path))
        volumes = sorted(os.listdir(volume_path))

        # loop through structure objects
        for com_file, origin_file, volume_file in zip(coms, origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(
                    f"{Path(origin_file).stem} and {Path(volume_file).stem} do not match"
                )
                sys.exit()
            structure = Path(origin_file).stem
            com_um = np.loadtxt(os.path.join(com_path, com_file))
            origin_allen = np.loadtxt(os.path.join(origin_path, origin_file))
            origin_um = origin_allen * self.um
            volume_allen = np.load(os.path.join(volume_path, volume_file))
            com_allen = center_of_mass(volume_allen)
            test_com = (origin_allen + com_allen) * self.um
            difference = com_um - test_com
            if structure == 'SC':
                print(f"{self.animal} {structure} com={np.round(com_um)}um origin um={np.round(origin_um)} COM={np.round(com_allen)} diff={difference}")

    @staticmethod
    def transform_create_alignment(points, transform):
        a = np.hstack((points, np.ones((points.shape[0], 1))))
        b = transform.T[:, 0:2]
        c = np.matmul(a, b)
        return c

    @staticmethod
    def create_volume_for_one_structure(polygons, pad_z):
        """Creates a volume from a dictionary of polygons
        The polygons are in the form of {section: [x,y]}
        """
        coords = list(polygons.values())
        min_vals, max_vals, mean_vals = get_min_max_mean(coords)
        if min_vals is None:
            return None, None
        min_x = min_vals[0]
        min_y = min_vals[1]
        max_x = max_vals[0]
        max_y = max_vals[1]
        xlength = max_x - min_x
        ylength = max_y - min_y
        slice_size = (int(round(ylength)), int(round(xlength)))
        #print(f'slice size={slice_size} {min_x=} {min_y=} {max_x=} {max_y=} {mean_vals=}', end=" ")
        volume = []
        # You need to subtract the min_x and min_y from the points as the volume is only as big as the range of x and y
        sections = []
        for section, points in sorted(polygons.items()):
            vertices = np.array(points) - np.array((min_x, min_y))
            volume_slice = np.zeros(slice_size, dtype=np.uint8)
            points = (vertices).astype(np.int32)
            cv2.fillPoly(volume_slice, pts=[points], color=255)
            volume.append(volume_slice)
            sections.append(int(section))

        volume = np.array(volume).astype(np.uint8)  # Keep this at uint8!
         # pad the volume in the z axis
        #volume = np.pad(volume, ((pad_z, pad_z), (0, 0), (0, 0)))  
        min_z = min(sections)
        #print(f"mean z = {np.mean(sections)}")
        origin = np.array([min_x, min_y, min_z]).astype(np.float64)
        return origin, volume


    @staticmethod
    def create_volume_for_one_structure_from_polygons(polygons, pad_z=0, transform=None):
        """Creates a volume from a dictionary of polygons
        The polygons are in the form of {section: [x,y]}
        """
        coords = list(polygons.values())
        min_vals, max_vals, mean_vals = get_min_max_mean(coords)
        if min_vals is None:
            return None, None
        min_x = min_vals[0]
        min_y = min_vals[1]
        max_x = max_vals[0]
        max_y = max_vals[1]
        min_z = int(min(polygons.keys()))
        max_z = int(max(polygons.keys()))  
        xlength = max_x - min_x
        ylength = max_y - min_y
        slice_size = (int(round(ylength)), int(round(xlength)))
        #print(f'slice size={slice_size} {min_x=} {min_y=} {max_x=} {max_y=} {mean_vals=} {min_z=} {max_z=}')
        volume = []
        # You need to subtract the min_x and min_y from the points as the volume is only as big as the range of x and y
        slices = []
        points_dict = {}
        for i, idx in enumerate(range(min_z, max_z)):
            volume_slice = np.zeros(slice_size, dtype=np.uint8)
            if idx in polygons:
                points = polygons[idx]
                points = np.array(points) - np.array((min_x, min_y))
                points = order_points_concave_hull(points, alpha=0.5)
                points = interpolate_points(points, 250)
                points = np.array(points).astype(np.int32)
                points_dict[i] = points
            else:
                try:
                    points = points_dict[i]
                except KeyError:
                    pass
        
            cv2.fillPoly(volume_slice, pts=[points], color=255)
            slices.append(volume_slice)
        if len(slices) == 0:
            return None, None
        volume = np.stack(slices, axis=0).astype(np.uint8)  # Keep this at uint8!
        ##### Transform volume with sitk, no translation
        origin = np.array([min_x, min_y, min_z]).astype(np.float64)
        if transform is not None:
            inverse_transform = transform.GetInverse()
            R = transform.GetParameters()[0:9]
            R_inv = inverse_transform.GetParameters()[0:9]
            translation = transform.GetParameters()[9:]
            inverse_translation = inverse_transform.GetParameters()[9:]
            del transform
            affine_transform = sitk.AffineTransform(3) 
            affine_transform.SetMatrix(R)
            affine_transform.SetTranslation((0,0,0))   
            
            resampler = sitk.ResampleImageFilter()
            # Set the transform
            resampler.SetTransform(affine_transform)
            # Set the reference image (determines output size, spacing, origin, and direction)
            # Often, the input image itself is used as the reference for the output grid.
            image = sitk.GetImageFromArray(volume.astype(np.float32))
            resampler.SetReferenceImage(image)
            # Set the interpolator (e.g., linear, nearest neighbor, B-spline)
            resampler.SetInterpolator(sitk.sitkLinear)
            # Set the default pixel value for areas outside the original image bounds
            resampler.SetDefaultPixelValue(0.0)
            # Execute the resampling
            resampled = resampler.Execute(image)
            volume = sitk.GetArrayFromImage(resampled)
            origin = origin + inverse_translation
        
        return origin, volume
    

    def fetch_create_volumes(self):    
        jsonpath = os.path.join(
            self.data_path, self.animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            sys.exit()
        with open(jsonpath) as f:
            aligned_dict = json.load(f)
        
        structures = list(aligned_dict.keys())
        for structure in structures:
            polygons = aligned_dict[structure]
            self.upsert_annotation(structure, polygons)

    def atlas2allen(self):
        atlas_all = list_coms(self.animal)
        allen_all = list_coms("Allen")
        common_keys = sorted(list(atlas_all.keys() & allen_all.keys()))
        good_keys = set(common_keys) - set(("RtTg", "AP"))
        atlas_src = np.array([atlas_all[s] for s in good_keys])
        allen_src = np.array([allen_all[s] for s in good_keys])
        matrix = compute_affine_transformation(atlas_src, allen_src)
        xy_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        polygons = defaultdict(list)
        structures = ['SC', 'IC', '7N_L', '7N_R']
        structures = ['SC']
        for structure in structures:
            label_ids = self.get_label_ids(structure)

            # Hard code annotator id to 1, Edward
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == self.animal)
                .filter(AnnotationSession.FK_user_id == 1)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .first()
            )
            if annotation_session is not None and annotation_session.annotation is not None:
                print(f"{annotation_session.id=}")
            else:
                print(f"No annotation session found for {self.animal} {structure}")
                continue

            annotation = annotation_session.annotation
            # first test data to make sure it has the right keys
            try:
                data = annotation["childJsons"]
            except KeyError as ke:
                print(f'No data for {annotation_session.FK_prep_id} was found. {ke}')
                return polygons
            
            for row in data:
                if 'childJsons' not in row:
                    return polygons
                for child in row['childJsons']:
                    x,y,z = child['pointA']
                    x *= M_UM_SCALE
                    y *= M_UM_SCALE
                    z *= M_UM_SCALE
                    # Get them in um for the transformation
                    x, y, z = affine_transform_point((x, y, z), matrix)
                    x = x/xy_resolution
                    y = y/xy_resolution
                    z = int(np.round(z/zresolution))
                    print(x,y,z)
                    polygons[z].append((x,y))

            if not self.debug:
                self.upsert_annotation(structure, polygons, get_even=True, animal='Allen')    

    def upsert_annotation(self, structure, polygons, get_even=True, animal=None):
        """Creates a volume from a dictionary of polygons
        The polygons are in the form of {section: [x,y]}  in the downsampled pixel space
        """
        xy_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution

        
        default_props = ["#ff0000", 1, 1, 5, 3, 1]

        reformatted_polygons = []
        centroids = []
        counter = 0
        test_redundant_z = []
        for z_index, (section, points) in enumerate(sorted(polygons.items())):
            section = int(section)
            if get_even:
                points = get_evenly_spaced_vertices(points)
            vertices = np.array(points)

            if len(vertices) == 0:
                continue
            new_lines = []
            new_polygon = {}
            parentAnnotationId = random_string()
            points = []
            for i in range(len(vertices)):
                try:
                    xa,ya = vertices[i]
                except ValueError:
                    continue
                try:
                    xb,yb = vertices[i+1]
                except IndexError:
                    xb,yb = vertices[0]
                except ValueError as ve:
                    print(f"Value Error B with {structure} {ve}")
                    continue

                xa = xa * xy_resolution * self.scaling_factor / M_UM_SCALE
                ya = ya * xy_resolution * self.scaling_factor / M_UM_SCALE
                # neuroglancer uses another 0.5 for the z axis
                z =  (section + 0.5) * zresolution / M_UM_SCALE
                xb = xb * xy_resolution * self.scaling_factor / M_UM_SCALE
                yb = yb * xy_resolution * self.scaling_factor / M_UM_SCALE

                pointA = [xa, ya, z]
                pointB = [xb, yb, z]
                
                new_line = {
                    "pointA": pointA,
                    "pointB": pointB,
                    "type": "line",
                    "parentAnnotationId": parentAnnotationId,
                    "props": default_props
                }
                new_lines.append(new_line)
                points.append(pointA)
                counter += 1

            current_z = section
            test_redundant_z.append(current_z)
            try:
                pre_z = test_redundant_z[z_index-1]
            except IndexError:
                pre_z = current_z
            
            if current_z != pre_z:
                # polygon keys
                parentAnnotationId = random_string()
                new_polygon["source"] = points[0]
                new_polygon["centroid"] = np.mean(points, axis=0).tolist()
                new_polygon["childrenVisible"] = True
                new_polygon["type"] = "polygon"
                new_polygon["parentAnnotationId"] = parentAnnotationId
                new_polygon["description"] = f"{structure}"
                new_polygon["props"] = default_props
                new_polygon["childJsons"] = new_lines

                centroids.append(new_polygon["centroid"])
                reformatted_polygons.append(new_polygon)

        # Create the annotation dictionary
        json_entry = {}
        json_entry["type"] = "volume"
        json_entry["props"] = default_props
        json_entry["source"] = centroids[0]
        json_entry["centroid"] = np.mean(centroids, axis=0).tolist()
        json_entry["childJsons"] = reformatted_polygons
        json_entry["description"] = f"{structure}"   

        if self.debug:
            centroid = json_entry["centroid"]
            print(f'total vertices={counter}', end=" ")
            print(f"len of centroids={len(centroids)} len of polygons={len(polygons)} len of reformatted_polygons={len(reformatted_polygons)}")

            xp,yp,zp = centroid
            xp *= M_UM_SCALE
            yp *= M_UM_SCALE
            zp *= M_UM_SCALE
            print(f"Centroid of {structure} {centroid=} xp={xp/xy_resolution}, yp={yp/xy_resolution}, zp={zp/zresolution}")
            return


        annotator_id = 1 # hard coded to Edward 
        label_ids = self.get_label_ids(structure)
        if animal is None:
            animal = self.animal

        if animal is None:
            print(f"animal is None")
            return
        
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == animal)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .filter(AnnotationSession.annotation["type"] == "volume")
                .one_or_none()
            )
        except Exception as e:
            print(f"Found more than one structure for {animal} {structure}. Exiting program, please fix")
            exit(1)

        if annotation_session is None:
            print(f"Inserting {structure} for {animal}")
            try:
                self.sqlController.insert_annotation_with_labels(
                    FK_user_id=annotator_id,
                    FK_prep_id=animal,
                    annotation=json_entry,
                    labels=[structure])
            except sqlalchemy.exc.OperationalError as e:
                print(f"Operational {e} for {structure}")
                self.sqlController.session.rollback()
        else:                
            update_dict = {'annotation': json_entry}
            print(f'Updating {animal} session {annotation_session.id} with {structure}')
            self.sqlController.update_session(annotation_session.id, update_dict=update_dict)


