import os
from pathlib import Path
import shutil
import sys
import numpy as np
from collections import defaultdict
import cv2
import json
from scipy.ndimage import center_of_mass
from sqlalchemy.exc import NoResultFound, MultipleResultsFound


from library.atlas.atlas_manager import AtlasToNeuroglancer
from library.atlas.atlas_utilities import apply_affine_transform, get_affine_transformation
from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationLabel, AnnotationSession
from library.image_manipulation.filelocation_manager import data_path, FileLocationManager
from library.registration.algorithm import umeyama
from library.utilities.atlas import volume_to_polygon, save_mesh, allen_structures
from library.utilities.utilities_process import SCALING_FACTOR, M_UM_SCALE, write_image


class BrainStructureManager():

    def __init__(self, animal, um=10, affine=False, debug=False):

        self.animal = animal
        self.fixed_brain = None
        self.sqlController = SqlController(animal)
        self.fileLocationManager = FileLocationManager(self.animal)
        self.data_path = os.path.join(data_path, 'atlas_data')
        self.volume_path = os.path.join(self.data_path, self.animal, 'structure')
        self.origin_path = os.path.join(self.data_path, self.animal, 'origin')
        self.mesh_path = os.path.join(self.data_path, self.animal, 'mesh')
        self.com_path = os.path.join(self.data_path, self.animal, 'com')
        self.point_path = os.path.join(self.fileLocationManager.prep, 'points')
        self.reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
        self.aligned_contours = {}
        self.com_annotator_id = 2
        self.polygon_annotator_id = 0
        self.debug = debug
        # self.midbrain_keys = {'SNC_L', 'SNC_R', 'SC', '3N_L', '3N_R', '4N_L', '4N_R', 'IC', 'PBG_L', 'PBG_R', 'SNR_L',  'SNR_R'}
        self.midbrain_keys = {'3N_L','3N_R','4N_L','4N_R','IC','PBG_L','PBG_R','SC','SNC_L','SNC_R','SNR_L','SNR_R'}
        self.allen_structures_keys = allen_structures.keys()
        self.allen_um = um # size in um of allen atlas
        self.com = None
        self.origin = None
        self.volume = None
        self.abbreviation = None

        self.affine = affine
        self.atlas_box_scales=(self.allen_um, self.allen_um, self.allen_um)
        self.atlas_raw_scale=10
        self.atlas_box_scales = np.array(self.atlas_box_scales)

        x_length = 1820
        y_length = 1000
        z_length = 1140
        self.atlas_box_size = np.array((x_length, y_length, z_length))
        self.atlas_box_center = self.atlas_box_size / 2

        os.makedirs(self.com_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.point_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)

    def load_aligned_contours(self):
        """load aligned contours
        """       
        aligned_and_padded_contour_path = os.path.join(self.data_path, self.animal, 'aligned_padded_structures.json')
        print(f'Loading JSON data from {aligned_and_padded_contour_path}')
        with open(aligned_and_padded_contour_path) as f:
            self.aligned_contours = json.load(f)

    def get_coms(self, annotator_id):
        """Get the center of mass values for this brain as an array

        Returns:
            np array: COM of the brain
        """
        # self.load_com()

        coms = self.sqlController.get_com_dictionary(self.animal, annotator_id=annotator_id)
        return coms

    def get_transform_to_align_brain(self, brain=None):
        """Used in aligning data to fixed brain
        TODO fix this to use the fixed brain
        """
        return np.eye(3), np.zeros((3,1))

        if brain.animal == self.fixed_brain.animal:
            return np.eye(3), np.zeros((3,1))

        moving_coms = brain.get_coms(brain.com_annotator_id)

        if 'midbrain' in brain.region:
            area_keys = self.midbrain_keys
        elif 'brainstem' in brain.region:
            area_keys = set(self.allen_structures_keys) - set(self.midbrain_keys)
        else:
            area_keys = moving_coms.keys()

        fixed_coms = self.fixed_brain.get_coms(annotator_id=self.fixed_brain.com_annotator_id)
        common_keys = sorted(fixed_coms.keys() & moving_coms.keys() & area_keys)
        fixed_points = np.array([fixed_coms[s] for s in common_keys])
        moving_points = np.array([moving_coms[s] for s in common_keys])

        # fixed_points /= self.allen_um
        # moving_points /= self.allen_um

        if fixed_points.shape != moving_points.shape or len(fixed_points.shape) != 2 or fixed_points.shape[0] < 3:
            print(f'Error calculating transform {brain.animal} {fixed_points.shape} {moving_points.shape} {common_keys}')
            print(f'Length fixed coms={len(fixed_coms.keys())} # moving coms={len(moving_coms.keys())}')
            sys.exit()
        print(f'In get transform and using moving shape={moving_points.shape} fixed shape={fixed_points.shape}')
        R, t = umeyama(moving_points.T, fixed_points.T)
        return R, t

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

    def compute_origin_and_volume_for_brain_structures(self, brainManager, brainMerger, polygon_annotator_id):
        """TODO this needs work. The volume has to be fetched from the new annotation session table instead
        of the polygon sequence table.
        """
        self.animal = brainManager.animal
        # controller = PolygonSequenceController(self.animal)
        # controller = AnnotationSessionController()
        # structures = controller.get_brain_regions()
        files = sorted(os.listdir(self.origin_path))
        structures = [str(structure).split('.')[0] for structure in files]
        # get transformation at um
        R, t = self.get_transform_to_align_brain(brainManager)
        if R is None:
            print(f'R is empty with {self.animal} ID={polygon_annotator_id}')
            return

        # loop through structure objects
        for structure in structures:
            self.abbreviation = structure
            if self.data_exists() or True:
                com_filepath = os.path.join(self.com_path, f'{self.abbreviation}.txt')
                if not os.path.exists(com_filepath):
                    print(f'{com_filepath} does not exist')
                    continue
                origin_filepath = os.path.join(self.origin_path, f'{self.abbreviation}.txt')
                volume_filepath = os.path.join(self.volume_path, f'{self.abbreviation}.npy')
                self.com = np.loadtxt(com_filepath)
                self.origin = np.loadtxt(origin_filepath)
                self.volume = np.load(volume_filepath)
            else:
                print('xxxxxxxxxxxxxxxxxxxxx')
                sys.exit()
                # if structure.abbreviation not in self.allen_structures_keys:
                #    continue
                df = controller.get_volume(self.animal, polygon_annotator_id, structure.id)
                if df.empty:
                    continue;

                #####TRANSFORMED point dictionary
                # polygons were drawn at xy resolution of 0.452um for the MDXXX brains
                # and 20um for the z axis
                polygons = defaultdict(list)

                for _, row in df.iterrows():
                    x = row['coordinate'][0] 
                    y = row['coordinate'][1] 
                    z = row['coordinate'][2]
                    print(f'structure={structure} x={x} y={y} z={z}')
                    # transform points to fixed brain um with rigid transform
                    # x,y,z = brain_to_atlas_transform((x,y,z), R, t)
                    # scale transformed points to self.allen_um.
                    x = x / SCALING_FACTOR / self.sqlController.scan_run.resolution
                    y = y / SCALING_FACTOR / self.sqlController.scan_run.resolution
                    z /= 20
                    xy = (x, y)
                    print(structure, x, y, z)
                    section = int(np.round(z))
                    polygons[section].append(xy)
                origin, section_size = self.get_origin_and_section_size(polygons)
                volume = []
                for _, contour_points in polygons.items():
                    # subtract origin so the array starts drawing in the upper top left
                    vertices = np.array(contour_points) - origin[:2]
                    contour_points = (vertices).astype(np.int32)
                    volume_slice = np.zeros(section_size, dtype=np.uint8)
                    cv2.drawContours(volume_slice, [contour_points], -1, (1), thickness=-1)
                    volume.append(volume_slice)
                volume = np.swapaxes(volume,0,2)
                # volume = gaussian(volume, 1)
                # set structure object values
                self.abbreviation = structure
                self.origin = origin
                self.volume = volume
                # Add origin and com
                self.com = np.array(self.get_center_of_mass()) + np.array(self.origin)
                # save individual structure, mesh and origin
                self.save_brain_origins_and_volumes_and_meshes()
                del origin, volume

            # merge data
            brainMerger.volumes_to_merge[structure].append(self.volume)
            brainMerger.origins_to_merge[structure].append(self.origin)
            brainMerger.coms_to_merge[structure].append(self.com)
            # debug info
            ids, counts = np.unique(self.volume, return_counts=True)
            print(polygon_annotator_id, self.animal, self.abbreviation, self.origin, self.com, end="\t")
            print(self.volume.dtype, self.volume.shape, end="\t")
            print(ids, counts)

    def inactivate_coms(self, animal):
        print('Inactivating COMS')
        return
        sessions = self.sqlController.get_active_animal_sessions(animal)
        for sc_session in sessions:
            sc_session.active=False
            controller.update_row(sc_session)

    def save_brain_origins_and_volumes_and_meshes(self):
        """Saves everything to disk, Except for the mesh, no calculations, only saving!
        """

        aligned_structure = volume_to_polygon(volume=self.volume, origin=self.origin, times_to_simplify=3)

        origin_filepath = os.path.join(self.origin_path, f'{self.abbreviation}.txt')
        volume_filepath = os.path.join(self.volume_path, f'{self.abbreviation}.npy')
        mesh_filepath = os.path.join(self.mesh_path, f'{self.abbreviation}.stl')
        com_filepath = os.path.join(self.com_path, f'{self.abbreviation}.txt')

        np.savetxt(origin_filepath, self.origin)
        np.save(volume_filepath, self.volume)
        save_mesh(aligned_structure, mesh_filepath)
        np.savetxt(com_filepath, self.com)

    def get_center_of_mass(self):
        com = center_of_mass(self.volume)
        sum_ = np.isnan(np.sum(com))
        if sum_:
            print(f'{self.animal} {self.abbreviation} has no COM {self.volume.shape} {self.volume.dtype} min={np.min(self.volume)} max={np.max(self.volume)}')
            ids, counts = np.unique(self.volume, return_counts=True)
            print(ids, counts)
            com = np.array([0,0,0])
        return com

    def data_exists(self):

        com_filepath = os.path.join(self.com_path, f'{self.abbreviation}.txt')
        origin_filepath = os.path.join(self.origin_path, f'{self.abbreviation}.txt')
        volume_filepath = os.path.join(self.volume_path, f'{self.abbreviation}.npy')

        com_exists = os.path.exists(com_filepath)
        origin_exists = os.path.exists(origin_filepath)
        volume_exists = os.path.exists(volume_filepath)

        if origin_exists and volume_exists and com_exists:
            return True
        else:
            return False

    def get_allen_id(self, structure: str) -> int:
        try:
            allen_color = allen_structures[structure]
        except KeyError:
            print(f'Could not get allen color for {structure}')
            allen_color = 0

        if type(allen_color) == list:
            allen_color = allen_color[0]

        return allen_color

    def update_database_com(self, structure:str, com: np.ndarray) -> None:
        """Annotator ID is hardcoded to 1
        Data coming in is in pixels, so we need to convert to um and then to meters
        """
        annotator_id = 1
        
        com = com.tolist()
        xy_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        x = com[0] * xy_resolution / M_UM_SCALE
        y = com[1] * xy_resolution / M_UM_SCALE
        z = com[2] * zresolution / M_UM_SCALE
        com = [x, y, z]
        json_entry = {"type": "point", "point": com, "description": f"{structure}", "centroid": com, "props": ["#ffff00", 1, 1, 5, 3, 1]}
        label = self.sqlController.get_annotation_label(structure)
        if label is not None:
            label_ids = [label.id]
        else:
            print(f'Could not find {structure} label in database')
            return
        # update label with allen ID
        allen_id = self.get_allen_id(structure=structure)
        label.allen_id = allen_id

        update_dict = {"allen_id": allen_id}
        self.sqlController.update_row(AnnotationLabel, label, update_dict=update_dict)

        
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == self.animal)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .filter(AnnotationSession.annotation['type'] == 'point')
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
            print(f'Multiple results found for {structure} error: {mrf}')
            return

        if annotation_session:
            print(f'Updating {structure} with {com} with ID={annotation_session.id}')
            update_dict = {"annotation": json_entry}
            self.sqlController.update_session(annotation_session.id, update_dict=update_dict)

    @staticmethod
    def check_for_existing_dir(path):
        if not os.path.exists(path):
            print(f'{path} does not exist, exiting.')
            sys.exit()

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
        """

        # origin is in animal scan_run.resolution coordinates
        # volume is in 10um
        self.check_for_existing_dir(self.origin_path)
        self.check_for_existing_dir(self.volume_path)

        atlas_volume = np.zeros((self.atlas_box_size), dtype=np.uint32)
        print(f'atlas box size={self.atlas_box_size} shape={atlas_volume.shape}')
        print(f'Using data from {self.origin_path}')
        origins = sorted(os.listdir(self.origin_path))
        volumes = sorted(os.listdir(self.volume_path))
        print(f'Working with {len(origins)} origins and {len(volumes)} volumes.')
        ids = {}
        atlas_centers = {}
        matrix = get_affine_transformation(self.animal)
        xs = []
        ys = []
        zs = []
        xsT = []
        ysT = []
        zsT = []

        for origin_file, volume_file in zip(origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(f'{Path(origin_file).stem} and {Path(volume_file).stem} do not match')
                sys.exit()
            structure = Path(origin_file).stem
            allen_color = self.get_allen_id(structure)
            ids[structure] = allen_color
            origin = np.loadtxt(os.path.join(self.origin_path, origin_file))
            volume = np.load(os.path.join(self.volume_path, volume_file))
            volume = np.rot90(volume, axes=(0, 1)) 
            volume = np.flip(volume, axis=0)
            # transform into the atlas box coordinates that neuroglancer assumes

            volume = volume * allen_color
            volume = volume.astype(np.uint32)
            volume[volume > 0] = allen_color
            volume = volume.astype(np.uint32)

            COM = center_of_mass(volume)
            center = (origin + COM )
            center = self.atlas_box_center + center * self.atlas_raw_scale / self.atlas_box_scales
            xs.append(center[0])
            ys.append(center[1])
            zs.append(center[2])
            if self.affine:
                print(f'Center of mass for {structure} is {center}', end="\t")
                center = apply_affine_transform(center, matrix)
                xsT.append(center[0])
                ysT.append(center[1])
                zsT.append(center[2])
                
                print(f'tranformed {center}')

            x_start = int(center[0] - COM[0])
            y_start = int(center[1] - COM[1])
            z_start = int(center[2] - COM[2])

            atlas_centers[structure] = center

            x_end = x_start + volume.shape[0]
            y_end = y_start + volume.shape[1]
            z_end = z_start + volume.shape[2]

            #print(f'Adding {structure} to atlas at {x_start}:{x_end} {y_start}:{y_end} {z_start}:{z_end}')

            try:
                atlas_volume[x_start:x_end, y_start:y_end, z_start:z_end] += volume            
            except ValueError as ve:
                print(f'Error adding {structure} to atlas: {ve}')
                continue
        if self.affine:
            print(f'Range of untransformed x={max(xs) - min(xs)} y={max(ys) - min(ys)} z={max(zs) - min(zs)}')
            print(f'Range of transformed x={max(xsT) - min(xsT)} y={max(ysT) - min(ysT)} z={max(zsT) - min(zsT)}')
            print(f'affine matrix=\n{matrix}')
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

        atlas_volume, atlas_centers, ids = self.create_atlas_volume()
        if self.debug:
            for k,v in atlas_centers.items():
                print(f'{k}={v}')   
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
            outpath = f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DKAtlas_{self.allen_um}um_sagittal.tif'

            if os.path.exists(outpath):
                print(f'Removing {outpath}')
                os.remove(outpath)
            print(f'saving image to {outpath}')
            write_image(outpath, atlas_volume)


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

        print(f'Pre Shape of atlas volume {atlas_volume.shape} dtype={atlas_volume.dtype}')


        if not self.debug:
            self.sqlController.session.close()
            atlas_name = f'DK.atlas.{self.allen_um}um'
            structure_path = os.path.join(self.data_path, 'structures', atlas_name)
            if os.path.exists(structure_path):
                print(f'Path exists: {structure_path}')
                print('Please remove before running this script')
                sys.exit(1)
            else:
                print(f'Creating data in {structure_path}')
            os.makedirs(structure_path, exist_ok=True)
            neuroglancer = AtlasToNeuroglancer(volume=atlas_volume, scales=self.atlas_box_scales * 1000)
            neuroglancer.init_precomputed(path=structure_path)
            neuroglancer.add_segment_properties(ids)
            neuroglancer.add_downsampled_volumes()
            neuroglancer.add_segmentation_mesh()


