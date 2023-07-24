import os
import numpy as np
from collections import defaultdict
import cv2
import json
from scipy.ndimage import center_of_mass


from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.controller.sql_controller import SqlController
from library.controller.structure_com_controller import StructureCOMController
from library.image_manipulation.filelocation_manager import data_path, FileLocationManager
from library.registration.algorithm import brain_to_atlas_transform, umeyama
from library.utilities.atlas import volume_to_polygon, save_mesh, allen_structures
from library.registration.volume_registration import VolumeRegistration
from library.controller.annotation_session_controller import AnnotationSessionController



class BrainStructureManager():

    def __init__(self, animal = 'MD589', atlas = 'atlasV8', debug=False):

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
        self.aligned_contours = {}
        self.annotator_id = 2
        self.volumeRegistration = VolumeRegistration(animal=self.animal)
        self.debug = debug
        #self.midbrain_keys = {'SC','IC','4N_L', '3N_L', 'PBG_L', '3N_R', 'PBG_R', '4N_R', 'SNR_L', 'SNR_R', 'Pn_L', 'Pn_R'}
        #self.midbrain_keys = {'IC','4N_L', '3N_L', 'PBG_L', '3N_R', 'PBG_R', '4N_R', 'SNR_L', 'SNR_R'}
        self.midbrain_keys = allen_structures.keys()
        self.allen_um = 25 # size in um of allen atlas
        
        self.com = None
        self.origin = None
        self.volume = None
        self.abbreviation = None

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



    def get_coms(self, annotator_id=2):
        """Get the center of mass values for this brain as an array

        Returns:
            np array: COM of the brain
        """
        #self.load_com()
        controller = StructureCOMController(self.animal)
        coms = controller.get_COM(self.animal, annotator_id=annotator_id)
        return coms

    def get_transform_to_align_brain(self, brain):
        """Used in aligning data to fixed brain
        """
        if brain.animal == self.fixed_brain.animal:
            return np.eye(3), np.zeros((3,1))

        moving_coms = brain.get_coms()
        fixed_coms = self.fixed_brain.get_coms(annotator_id=self.fixed_brain.annotator_id)
        common_keys = fixed_coms.keys() & moving_coms.keys() & self.midbrain_keys
        brain_regions = sorted(moving_coms.keys())

        fixed_points = np.array([fixed_coms[s] for s in brain_regions if s in common_keys])
        moving_points = np.array([moving_coms[s] for s in brain_regions if s in common_keys])

        fixed_points /= 25
        moving_points /= 25

        if fixed_points.shape != moving_points.shape or len(fixed_points.shape) != 2 or fixed_points.shape[0] < 3:
            print(brain.animal, fixed_points.shape, moving_points.shape, common_keys)
            return None, None

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
        section_size = np.array([xspan, yspan]).astype(int)
        return origin, section_size


    def compute_origin_and_volume_for_brain_structures(self, brainManager, brainMerger, annotator_id):
        self.animal = brainManager.animal
        polygon = PolygonSequenceController(animal=self.animal)
        controller = StructureCOMController(self.animal)
        structures = controller.get_structures()
        # get transformation at um 
        
        R, t = self.get_transform_to_align_brain(brainManager)
        if R is None:
            return
        
        # loop through structure objects
        for structure in structures:
            structure.abbreviation
            if structure.abbreviation not in self.midbrain_keys:
                continue

            if structure.abbreviation in ['Pn_L', 'Pn_R']:
                continue

            df = polygon.get_volume(self.animal, annotator_id, structure.id)
            if df.empty:
                continue;

            #####TRANSFORMIX points = []
            polygons = defaultdict(list)

            for _, row in df.iterrows():
                x = row['coordinate'][0] 
                y = row['coordinate'][1] 
                z = row['coordinate'][2]
                # transform points to fixed brain um with rigid transform

                # scale transformed points to 25um
                x /= self.allen_um
                y /= self.allen_um
                z /= self.allen_um
                x,y,z = brain_to_atlas_transform((x,y,z), R, t)
                #####TRANSFORMIX points.append((x,y,z))
                xy = (x, y)
                section = int(np.round(z))
                polygons[section].append(xy)

            color = 1 # on/off
            origin, section_size = self.get_origin_and_section_size(polygons)
            volume = []
            for _, contour_points in polygons.items():
                vertices = np.array(contour_points)
                # subtract origin so the array starts drawing in the upper top left
                vertices = np.array(contour_points) - origin[:2]
                contour_points = (vertices).astype(np.int32)
                volume_slice = np.zeros(section_size, dtype=np.uint8)
                volume_slice = cv2.polylines(volume_slice, [contour_points], isClosed=True, color=color, thickness=1)
                volume_slice = cv2.fillPoly(volume_slice, pts=[contour_points], color=color)
                volume.append(volume_slice)
            volume = np.array(volume).astype(np.bool8)
            volume = np.swapaxes(volume,0,2)
            # set structure object values
            self.abbreviation = structure.abbreviation
            self.origin = origin
            self.volume = volume
            # Add origin and com
            self.com = np.array(self.get_center_of_mass()) + np.array(self.origin)
            brainMerger.volumes_to_merge[structure.abbreviation].append(self.volume)
            brainMerger.origins_to_merge[structure.abbreviation].append(self.origin)
            brainMerger.coms_to_merge[structure.abbreviation].append(self.com)
            # debug info
            ids, counts = np.unique(volume, return_counts=True)
            print(annotator_id, self.animal, self.abbreviation, self.origin, self.com, end="\t")
            print(volume.dtype, volume.shape, end="\t")
            print(ids, counts)


    def update_com(self, com, structure_id):
        source = "MANUAL"
        controller = AnnotationSessionController(self.animal)
        annotation_session = controller.get_annotation_session(self.animal, structure_id, 2)
        x = com[0] * 25
        y = com[1] * 25
        z = com[2] * 25
        entry = {'source': source, 'FK_session_id': annotation_session.id, 'x': x, 'y':y, 'z': z}
        controller.upsert_structure_com(entry)

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
        

    def transformix_points(self, points):
        polygons = defaultdict(list)

        write_filepath = os.path.join(self.point_path, 'point.pts')
        transformixImageFilter = self.volumeRegistration.setup_transformix(
            self.volumeRegistration.reverse_elastix_output)

        # First, write all data to a pts file
        with open(write_filepath, 'w') as f:
            f.write('point\n')
            f.write(f'{len(points)}\n')
            for point in points:
                x = point[0]
                y = point[1]
                z = point[2]
                f.write(f'{x} {y} {z}')
                f.write('\n')

        # done writing, now transform
        transformixImageFilter.SetFixedPointSetFileName(write_filepath)
        transformixImageFilter.Execute()
        # done transforming, now read
        read_filepath = self.volumeRegistration.registered_point_file
        with open(read_filepath, "r") as f:
            lines = f.readlines()
            f.close()
        # done reading, now stuff into dictionary and return
        point_or_index = 'OutputPoint'
        for i in range(len(lines)):
            lx = lines[i].split()[lines[i].split().index(
                point_or_index)+3:lines[i].split().index(point_or_index)+6]  # x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]

            xy = (x, y)
            section = int(np.round(z))
            polygons[section].append(xy)

        return polygons


    def get_center_of_mass(self):
        com = center_of_mass(self.volume)
        sum_ = np.isnan(np.sum(com))
        if sum_:
            print(f'{self.animal} {self.abbreviation} has no COM {self.volume.shape} {self.volume.dtype} min={np.min(self.volume)} max={np.max(self.volume)}')
            ids, counts = np.unique(self.volume, return_counts=True)
            print(ids, counts)
            com = np.array([0,0,0])
        return com