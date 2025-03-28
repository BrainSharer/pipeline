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
from scipy.ndimage import center_of_mass

from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from tqdm import tqdm

from library.image_manipulation.pipeline_process import Pipeline
from library.atlas.atlas_manager import AtlasToNeuroglancer
from library.atlas.atlas_utilities import (
    adjust_volume,
    apply_affine_transform,
    compute_affine_transformation,
    get_affine_transformation,
    get_min_max_mean,
    list_coms,
    ORIGINAL_ATLAS,
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
    get_image_size,
    read_image,
    write_image,
)
from library.utilities.utilities_contour import get_contours_from_annotations


class BrainStructureManager:

    def __init__(self, animal, um=10, affine=False, debug=False):

        self.animal = animal
        self.fixed_brain = None
        self.sqlController = SqlController(animal)
        self.fileLocationManager = FileLocationManager(self.animal)
        self.data_path = os.path.join(data_path, "atlas_data")
        self.structure_path = os.path.join(data_path, "pipeline_data", "structures")

        self.com_path = os.path.join(
            self.data_path, self.animal, "com"
        )  # need this for the transforamtion matrix
        self.origin_path = os.path.join(self.data_path, self.animal, "origin")
        self.mesh_path = os.path.join(self.data_path, self.animal, "mesh")
        self.volume_path = os.path.join(self.data_path, self.animal, "structure")

        self.aligned_contours = {}
        self.com_annotator_id = 2
        self.polygon_annotator_id = 0
        self.debug = debug
        self.allen_um = um  # size in um of allen atlas
        self.com = None
        self.origin = None
        self.volume = None
        self.abbreviation = None

        self.affine = affine
        self.atlas_box_scales = np.array((self.allen_um, self.allen_um, self.allen_um))
        # self.atlas_box_scales = np.array((14.464, 14.464, 20))
        self.atlas_raw_scale = 10

        self.allen_x_length = 1820
        self.allen_y_length = 1000
        self.allen_z_length = 1140
        #unpadded_allen_x_length = 1320
        #unpadded_allen_y_length = 800
        #unpadded_allen_z_length = 1140
        self.atlas_box_size = np.array(
            (self.allen_x_length, self.allen_y_length, self.allen_z_length)
        )
        self.atlas_box_center = self.atlas_box_size / 2

        self.allen_resolution = np.array([10, 10, 10])
        self.atlas2allen = (
            np.array((0.452 * 32, 0.452 * 32, 20)) / self.allen_resolution
        )

        os.makedirs(self.com_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)

    @staticmethod
    def get_transform_to_align_brain(moving_brain, annotator_id=2):
        """Transform x,y,z points to Atlas space."""

        fixed_brain = "AtlasV8"
        moving_coms = list_coms(moving_brain, annotator_id=annotator_id)
        fixed_coms = list_coms(fixed_brain, annotator_id=annotator_id)

        for structure, com in moving_coms.items():
            x, y, z = com
            moving_coms[structure] = [x, y, z]

        common_keys = sorted(list(fixed_coms.keys() & moving_coms.keys()))
        fixed_points = np.array([fixed_coms[s] for s in common_keys])
        moving_points = np.array([moving_coms[s] for s in common_keys])

        if len(fixed_points) < 3 or len(moving_points) < 3:
            print(f"Not enough points to align {moving_brain} to {fixed_brain}")
            return np.eye(4)

        transformation_matrix = compute_affine_transformation(
            moving_points, fixed_points
        )

        return transformation_matrix

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

    def create_brains_origin_volume(self, brainMerger, animal, fixed_brain):
        """ """
        self.animal = animal
        origin_path = os.path.join(self.data_path, self.animal, "origin")
        volume_path = os.path.join(self.data_path, self.animal, "structure")
        self.check_for_existing_dir(origin_path)
        self.check_for_existing_dir(volume_path)
        origins = sorted(os.listdir(origin_path))
        volumes = sorted(os.listdir(volume_path))

        # loop through structure objects
        for origin_file, volume_file in zip(origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(
                    f"{Path(origin_file).stem} and {Path(volume_file).stem} do not match"
                )
                sys.exit()
            structure = Path(origin_file).stem
            # if structure not in ['IC', 'VLL_L', 'VLL_R']:
            #    continue

            self.origin = np.loadtxt(os.path.join(origin_path, origin_file))
            self.volume = np.load(os.path.join(volume_path, volume_file))

            if self.debug:
                print(f"{animal} {structure} origin={np.round(self.origin)}")
            else:
                # merge data
                brainMerger.origins_to_merge[structure].append(self.origin)
                brainMerger.volumes_to_merge[structure].append(self.volume)


    def create_brains_origin_volume_from_polygons(self, brainMerger, structure, debug=False):

        label = self.sqlController.get_annotation_label(structure)
        if label is not None:
            label_ids = [label.id]
        else:
            print(f"Could not find {structure} label in database")
            return

        annotation_sessions = (
            self.sqlController.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
            .all()
        )

        for annotation_session in annotation_sessions:
            animal = annotation_session.FK_prep_id

            """
            transformation_matrix = self.get_transform_to_align_brain(
                moving_brain=animal,
                annotator_id=2,
            )
            """

            polygons = self.sqlController.get_annotation_volume(annotation_session.id)
            if len(polygons) < 50:
                continue
            origin, volume = self.create_volume_for_one_structure(polygons)


            if origin is None or volume is None:
                print(f"{structure} {annotation_session.FK_prep_id} has no volumes to merge")
                return None
            
            volume = np.swapaxes(volume, 0, 2)

            if debug:
                print(f"Adding {structure=} to {animal=} with {origin=} and {volume.shape=} len polygons={len(polygons)}")
            else:
                brainMerger.origins_to_merge[structure].append(origin)
                brainMerger.volumes_to_merge[structure].append(volume)

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

    def update_database_com(self, structure: str, com: np.ndarray) -> None:
        """Annotator ID is hardcoded to 1
        Data coming in is in pixels, so we need to convert to um and then to meters
        """
        annotator_id = 1

        com = com.tolist()
        # xy_resolution = self.sqlController.scan_run.resolution
        # zresolution = self.sqlController.scan_run.zresolution
        xy_resolution = 10
        zresolution = 10
        x = com[0] * xy_resolution / M_UM_SCALE
        y = com[1] * xy_resolution / M_UM_SCALE
        z = com[2] * zresolution / M_UM_SCALE
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
        if False:
            allen_id = self.get_allen_id(structure=structure)
            label.allen_id = allen_id

            update_dict = {"allen_id": allen_id}
            self.sqlController.update_row(
                AnnotationLabel, label, update_dict=update_dict
            )

        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == self.animal)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .filter(AnnotationSession.annotation["type"] == "point")
                .one()
            )
        except NoResultFound as nrf:
            print(f"Inserting {structure} with {com}")
            self.sqlController.insert_annotation_with_labels(
                FK_user_id=annotator_id,
                FK_prep_id=self.animal,
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
        new_point = apply_affine_transform(point)
        return new_point

    def create_atlas_volume(self):
        """
        Creates an atlas volume by combining individual brain structure volumes.

        This method reads origin and volume data for various brain structures, transforms
        the volumes into the atlas coordinate system, and combines them into a single
        atlas volume. The atlas volume is represented as a 3D numpy array where each
        voxel value corresponds to a specific brain structure.

        Returns:
            tuple: A tuple containing:
                - atlas_volume (np.ndarray): The combined atlas volume.
                - atlas_centers (dict): A dictionary mapping each brain structure to its center coordinates in the atlas.
                - ids (dict): A dictionary mapping each brain structure to its corresponding Allen Institute color ID.

        Raises:
            SystemExit: If the origin and volume filenames do not match.
            ValueError: If there is an error adding a structure to the atlas volume.

        Notes:
            - The origin and volume data are expected to be in specific directories defined by `self.origin_path` and `self.volume_path`.
            - The method assumes that the filenames in the origin and volume directories match.
            - The volumes are transformed and scaled according to the atlas box coordinates and scales.
            - The method uses an affine transformation if `use_transformed` is set to True.

            for the foundation brains:
            scales = np.array([0.452*32, 0.452*32, 20])

        """

        # origin is in animal scan_run.resolution coordinates
        # volume is in 10um
        self.check_for_existing_dir(self.origin_path)
        self.check_for_existing_dir(self.volume_path)

        atlas_volume = np.zeros((self.atlas_box_size), dtype=np.uint32)
        print(f"atlas box size={self.atlas_box_size} shape={atlas_volume.shape}")
        print(f"Using data from {self.origin_path}")
        origins = sorted(os.listdir(self.origin_path))
        volumes = sorted(os.listdir(self.volume_path))
        print(f"Working with {len(origins)} origins and {len(volumes)} volumes.")
        ids = {}
        atlas_centers = {}
        transformation_matrix = get_affine_transformation(self.animal)
        translations = transformation_matrix[..., 3]
        scaled_translations = translations / np.hstack((self.atlas_box_scales, 1))
        transformation_matrix[..., 3] = scaled_translations.T
        upper = 150

        for origin_file, volume_file in zip(origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(
                    f"{Path(origin_file).stem} and {Path(volume_file).stem} do not match"
                )
                sys.exit()
            structure = Path(origin_file).stem
            allen_id = self.get_allen_id(structure)
            try:
                ids[structure] = allen_id
            except IndexError as ke:
                print(f"Problem with index error: {structure=} {allen_id=} in database")
                sys.exit()
            origin = np.loadtxt(os.path.join(self.origin_path, origin_file))
            volume = np.load(os.path.join(self.volume_path, volume_file))
            if self.animal == ORIGINAL_ATLAS:
                volume = np.rot90(volume, axes=(0, 1))
                volume = np.flip(volume, axis=0)

            COM = center_of_mass(volume)
            volume = adjust_volume(volume, allen_id)

            if math.isnan(COM[0]):
                nids, ncounts = np.unique(volume, return_counts=True)
                print(f"{structure} volume is invalid {nids} {ncounts}")
                continue
                COM = (0, 0, 0)
            # transform into the atlas box coordinates that neuroglancer assumes
            origin_and_com = origin + COM
            center = (
                self.atlas_box_center
                + origin_and_com * self.atlas_raw_scale / self.atlas_box_scales
            )
            atlas_centers[structure] = center
            if self.affine:
                center = apply_affine_transform(center, transformation_matrix)

            x_start = int(center[0] - COM[0])
            y_start = int(center[1] - COM[1])
            z_start = int(center[2] - COM[2])
            # x_start = int(x) + self.allen_x_length // 2
            # y_start = int(y) + self.allen_y_length // 2
            # z_start = int(z) + self.allen_z_length // 2

            x_end = x_start + volume.shape[0]
            y_end = y_start + volume.shape[1]
            z_end = z_start + volume.shape[2]

            if self.debug:
                print(
                    f"{structure} center={np.round(center)} x={x_start}:{x_end} y={y_start}:{y_end} z={z_start}:{z_end}"
                )
            else:
                try:
                    atlas_volume[x_start:x_end, y_start:y_end, z_start:z_end] += volume
                except ValueError as ve:
                    print(f"Error adding {structure} to atlas: {ve}")
                    continue

        if self.affine:
            print(f"Transformation matrix\n {transformation_matrix}")

        return atlas_volume, atlas_centers, ids

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

        print(f"evaluating atlas data from {self.com_path}")
        atlas_volume, atlas_centers, ids = self.create_atlas_volume()

        if self.debug:
            for k, v in atlas_centers.items():
                print(f"{k}={v}")
        else:
            for structure, com in atlas_centers.items():
                self.update_database_com(structure, com)

    def save_atlas_volume(self) -> None:
        """
        Saves the atlas volume to a specified file path.

        This method creates an atlas volume and saves it to a predefined location.
        If the file already exists at the specified location, it will be removed
        before saving the new atlas volume.

        Returns:
            None
        """

        atlas_volume, atlas_centers, ids = self.create_atlas_volume()
        if not self.debug:
            outpath = f"/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK{self.animal}_{self.allen_um}um_sagittal.tif"
            midpath = f"/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK{self.animal}_{self.allen_um}um_midpath.tif"

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

    def create_neuroglancer_volume(self):
        """
        Creates a Neuroglancer volume from the atlas volume.

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

        atlas_volume, atlas_centers, ids = self.create_atlas_volume()

        print(
            f"Pre Shape of atlas volume {atlas_volume.shape} dtype={atlas_volume.dtype}"
        )

        if not self.debug:
            aligned = "aligned" if self.affine else "unaligned"
            atlas_name = f"DK.{self.animal}.{aligned}.{self.allen_um}um"
            structure_path = os.path.join(self.structure_path, atlas_name)
            if os.path.exists(structure_path):
                print(f"Path exists: {structure_path}")
                print("Please remove before running this script")
                sys.exit(1)
            else:
                print(f"Creating data in {structure_path}")
            os.makedirs(structure_path, exist_ok=True)
            neuroglancer = AtlasToNeuroglancer(
                volume=atlas_volume, scales=self.atlas_box_scales * 1000
            )
            neuroglancer.init_precomputed(path=structure_path)
            neuroglancer.add_segment_properties(ids)
            neuroglancer.add_downsampled_volumes()
            neuroglancer.add_segmentation_mesh()

    def create_brain_volumes_origins(self, brainMerger, animal, debug):
        """Step 2"""

        jsonpath = os.path.join(
            self.data_path, animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            sys.exit()
        with open(jsonpath) as f:
            aligned_dict = json.load(f)
        structures = list(aligned_dict.keys())
        desc = f"Create {animal} coms/meshes/origins/volumes"
        for structure in tqdm(structures, desc=desc, disable=debug):
            polygons = aligned_dict[structure]
            origin, volume = self.create_volume_for_one_structure(polygons)
            if debug:
                print(f"Adding {structure} to {animal} with {np.round(origin)} and {volume.shape=} len polygons={len(polygons)}")
            else:
                brainMerger.volumes[structure] = volume
                brainMerger.origins[structure] = origin

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
                            img, [points], isClosed=True, color=1, thickness=5
                        )

            write_image(outfile, img)

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

                points = np.array(section_structure_vertices[section][structure]) / 32
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
        errors = []
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
                error = f"{structure} has only one value {nids} {ncounts} {com}, please check\n"
                errors.append(error)
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
            transformed = apply_affine_transform(atlas0, matrix)
            transformed = [x for x in transformed]
            difference = [a - b for a, b in zip(transformed, allen0)]
            ss = sum_square_com(difference)
            error.append(ss)
            print(
                f"{structure} atlas={np.round(atlas0)} allen={np.round(allen0)} transformed={np.round( np.array(transformed)) } \
                error={np.round(np.array(difference))} ss={round(ss,2)}"
            )
        print("RMS", sum(error) / len(common_keys))

    @staticmethod
    def transform_create_alignment(points, transform):
        a = np.hstack((points, np.ones((points.shape[0], 1))))
        b = transform.T[:, 0:2]
        c = np.matmul(a, b)
        return c

    @staticmethod
    def create_volume_for_one_structure(polygons):
        """Creates a volume from a dictionary of polygons
        The polygons are in the form of {section: [x,y]}
        """
        coords = list(polygons.values())
        min_vals, max_vals, mean_vals = get_min_max_mean(coords)
        if min_vals is None:
            return None, None
        min_x = min_vals[0]
        min_y = min_vals[1]
        ylength = max_vals[1] - min_vals[1]
        xlength = max_vals[0] - min_vals[0]
        slice_size = (int(round(ylength)), int(round(xlength)))
        volume = []
        # You need to subtract the min_x and min_y from the points as the volume is only as big as the range of x and y
        sections = []
        for section, points in sorted(polygons.items()):
            vertices = np.array(points) - np.array((min_x, min_y))
            volume_slice = np.zeros(slice_size, dtype=np.uint8)
            points = (vertices).astype(np.int32)
            volume_slice = cv2.fillPoly(volume_slice, pts=[points], color=255)
            volume.append(volume_slice)
            sections.append(section)

        volume = np.array(volume).astype(np.uint8)  # Keep this at uint8!
        pad_z = 40 # pad the volume in the z axis
        volume = np.pad(volume, ((pad_z, pad_z), (0, 0), (0, 0)))  

        min_z = min(sections)
        # fix x and y so they get the correct origin
        min_x += xlength
        min_y += ylength
        origin = np.array([min_x, min_y, min_z]).astype(np.float64)
        return origin, volume
