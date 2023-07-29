"""
This script will take a source brain (where the data comes from) and an image brain 
(the brain whose images you want to display unstriped) and align the data from the point brain
to the image brain. It first aligns the point brain data to the atlas, then that data
to the image brain. It prints out the data by default and also will insert
into the database if given a layer name.
"""
import os
import numpy as np
from collections import defaultdict
from skimage.filters import gaussian
import pandas as pd

from library.image_manipulation.filelocation_manager import data_path
from library.registration.algorithm import brain_to_atlas_transform, umeyama
from library.utilities.atlas import volume_to_polygon, save_mesh
from library.utilities.atlas import singular_structures
from library.registration.brain_structure_manager import BrainStructureManager
from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import AnnotationType, StructureCOM


class BrainMerger():

    def __init__(self, debug=False):
        self.symmetry_list = singular_structures
        self.volumes_to_merge = defaultdict(list)
        self.origins_to_merge = defaultdict(list)
        self.coms_to_merge = defaultdict(list)
        atlas = 'atlasV8'
        self.data_path = os.path.join(data_path, 'atlas_data', atlas)
        self.volume_path = os.path.join(self.data_path, 'structure')
        self.origin_path = os.path.join(self.data_path, 'origin')
        self.mesh_path = os.path.join(self.data_path, 'mesh')
        self.csv_path = os.path.join(self.data_path, 'csv')
        self.volumes = {}
        self.coms = {}
        self.origins = {}
        self.com_path = os.path.join(self.data_path, 'com')
        self.margin = 50
        self.threshold = 0.25  # the closer to zero, the bigger the structures
        # a value of 0.01 results in very big close fitting structures

        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.com_path, exist_ok=True)
        os.makedirs(self.csv_path, exist_ok=True)


    def pad_volume(self, size, volume):
        size_difference = size - volume.shape
        xr, yr, zr = ((size_difference)/2).astype(int)
        xl, yl, zl = size_difference - np.array([xr, yr, zr])
        return np.pad(volume, [[xl, xr], [yl, yr], [zl, zr]])

    def get_merged_landmark_probability(self, structure, volumes):
        lvolumes = len(volumes)
        if lvolumes == 1:
            print(f'{structure} has only one volume')
            return volumes[0]
        elif lvolumes > 1:
            sizes = np.array([vi.shape for vi in volumes])
            volume_size = sizes.max(0) + self.margin
            volumes = [self.pad_volume(volume_size, vi) for vi in volumes]
            volumes = list([(v > 0).astype(np.int32) for v in volumes])

            merged_volume = np.sum(volumes, axis=0)
            merged_volume_prob = merged_volume / float(np.max(merged_volume))
            # increasing the STD makes the volume smoother
            # Smooth the probability
            average_volume = gaussian(merged_volume_prob, 3.0)
            color = 1
            average_volume[average_volume > self.threshold] = color
            average_volume[average_volume != color] = 0
            average_volume = average_volume.astype(np.uint8)
            return average_volume
        else:
            print(f'{structure} has no volumes to merge')
            return None

    def save_atlas_origins_and_volumes_and_meshes(self):
        """Saves everything to disk
        """

        origins = {structure: np.mean(origin, axis=0) for structure, origin in self.origins_to_merge.items()}
        coms = {structure: np.mean(com, axis=0) for structure, com in self.coms_to_merge.items()}
        #origins = self.transform_origins(origins)

        for structure in self.volumes.keys():
            volume = self.volumes[structure]
            origin = origins[structure]
            com = coms[structure]
            # stuff self.coms and self.origins to save to DB later.
            self.coms[structure] = com
            self.origins[structure] = origin
            # mesh needs a center in the middle for all the STL files
            origins_array = np.array(list(origins.values()))
            mesh_origin = origin - origins_array.mean(0)
            aligned_structure = volume_to_polygon(volume=volume, origin=mesh_origin, times_to_simplify=3)
            
            origin_filepath = os.path.join(self.origin_path, f'{structure}.txt')
            volume_filepath = os.path.join(self.volume_path, f'{structure}.npy')
            mesh_filepath = os.path.join(self.mesh_path, f'{structure}.stl')
            com_filepath = os.path.join(self.com_path, f'{structure}.txt')

            np.savetxt(origin_filepath, origin)
            np.save(volume_filepath, volume)
            save_mesh(aligned_structure, mesh_filepath)
            np.savetxt(com_filepath, com)


    def save_coms_to_db(self):
        """Saves COMs to DB
        """
        animal = 'Atlas'
        brainManager = BrainStructureManager(animal)
        source = 'MANUAL'
        brainManager.inactivate_coms(animal)
        structureController = StructureCOMController(animal)
        annotationSessionController = AnnotationSessionController(animal)

        for abbreviation in self.coms.keys():
            point = self.coms[abbreviation]
            #origin = self.origins[abbreviation]
            FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
            FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.STRUCTURE_COM, 
                                                                                    FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
            x,y,z = (p*25 for p in point)
            #minx, miny, minz = (p for p in origin)
            com = StructureCOM(source=source, x=x, y=y, z=z, FK_session_id=FK_session_id)
            brainManager.sqlController.add_row(com)


    # should be static
    def calculate_distance(self, com1, com2):
        return (np.linalg.norm(com1 - com2))

    # should be static
    def label_left_right(self, row):
        val = 'Singular'
        if str(row['Structure']).endswith('L'):
            val = 'Left'
        if str(row['Structure']).endswith('R'):
            val = 'Right'
        return val        


    def transform_origins(self, moving_origins):
        """moving origins VS fixed COMs
        fixed_coms have to be adjusted to subtract the COM
        so they become origins
        """
        fixed_brain = BrainStructureManager('Allen')
        fixed_coms = fixed_brain.get_coms(annotator_id=1)

        common_keys = fixed_coms.keys() & moving_origins.keys()
        brain_regions = sorted(moving_origins.keys())

        """
        #fixed_points_list = []
        fixed_volume_list = []
        fixed_volume_dict = {}
        for structure in brain_regions:
            if structure in common_keys:
                volume_com = center_of_mass(self.volumes[structure])
                sum_ = np.isnan(np.sum(volume_com))
                if sum_:
                    v = self.volumes[structure]
                    print(f'{structure} has no COM {v.shape} {v.dtype} min={np.min(v)} max={np.max(v)}')
                    ids, counts = np.unique(v, return_counts=True)
                    print(ids, counts)
                    volume_com = np.array([0,0,0])
                fixed_volume_dict[structure] = volume_com
        """
        
        fixed_points = np.array([fixed_coms[s] for s in brain_regions if s in common_keys]) / 25
        moving_points = np.array([moving_origins[s] for s in brain_regions if s in common_keys])
        fixed_point_dict = {s:fixed_coms[s] for s in brain_regions if s in common_keys}
        moving_point_dict = {s:moving_origins[s] for s in brain_regions if s in common_keys}

        assert fixed_points.shape == moving_points.shape, 'Shapes do not match'
        assert len(fixed_points.shape) == 2, f'Dimensions are wrong fixed shape={fixed_points.shape} commonkeys={common_keys}'
        assert fixed_points.shape[0] > 2, 'Not enough points'

        transformed_origins = {}
        R, t = umeyama(moving_points.T, fixed_points.T)
        for structure, origin in moving_origins.items():
            point = brain_to_atlas_transform(origin, R, t)
            transformed_origins[structure] = point


        distances = []
        for structure in common_keys:
            (x,y,z) = fixed_point_dict[structure]
            fixed_point = np.array([x,y,z])    
            moving_point = np.array(moving_point_dict[structure])
            reg_point = brain_to_atlas_transform(moving_point, R, t)
            d = self.calculate_distance(fixed_point, reg_point)
            distances.append(d)
            print(f'{structure} distance={round(d,2)}')
        
        print(f'length={len(distances)} mean={round(np.mean(distances))} min={round(min(distances))} max={round(max(distances))}')

        return transformed_origins

    def evaluate(self):
        annotator_id = 1 # Edward created all the COMs for the DK atlas and the Allen
        animal = 'Atlas'
        brainManager = BrainStructureManager(animal)
        brainManager.com_annotator_id = annotator_id
        brainManager.fixed_brain = BrainStructureManager('Allen')
        brainManager.fixed_brain.com_annotator_id = annotator_id

        R, t = brainManager.get_transform_to_align_brain(brainManager)

        atlas_coms = brainManager.get_coms(annotator_id=annotator_id)
        allen_coms = brainManager.fixed_brain.get_coms(annotator_id=annotator_id)
        common_keys = allen_coms.keys() & atlas_coms.keys()
        brain_regions = sorted(atlas_coms.keys())
        allen_point_dict = {s:allen_coms[s] for s in brain_regions if s in common_keys}
        atlas_point_dict = {s:atlas_coms[s] for s in brain_regions if s in common_keys}

        distances = []
        sortme = {}
        for structure in common_keys:
            (xallen, yallen, zallen) = allen_point_dict[structure]
            (xatlas, yatlas, zatlas) = atlas_point_dict[structure]
            (xreg, yreg, zreg) = brain_to_atlas_transform((xatlas, yatlas, zatlas), R, t)
            allen_point = np.array([xallen, yallen, zallen])
            reg_point = np.array([xreg, yreg, zreg])
            d = self.calculate_distance(allen_point, reg_point)
            distances.append(d)
            sortme[structure] = d

        print(f'n={len(distances)}, min={min(distances)} max={max(distances)}, mean={np.mean(distances)}')

        ds = {k: v for k, v in sorted(sortme.items(), key=lambda item: item[1])}
        for structure, d in ds.items():
            print(f'{structure} distance from Allen={round(d,2)} micrometers')
            

    def save_brain_area_data(self):
        animal = 'Atlas'
        brain = BrainStructureManager(animal)
        brain.fixed_brain = BrainStructureManager('Allen')
        if brain.midbrain:
            csvfile = "midbrain"
            area_keys = brain.midbrain_keys
        else:
            csvfile = "brainstem"
            area_keys = set(brain.allen_structures_keys) - brain.midbrain_keys 
        moving_coms = brain.get_coms(annotator_id=1)
        allen_coms = brain.fixed_brain.get_coms(annotator_id=1)
        common_keys = allen_coms.keys() & moving_coms.keys() & area_keys
        brain_regions = sorted(moving_coms.keys())
        allen_points = np.array([allen_coms[s] for s in brain_regions if s in common_keys])
        moving_points = np.array([moving_coms[s] for s in brain_regions if s in common_keys])
        allen_point_dict = {s:allen_coms[s] for s in brain_regions if s in common_keys}
        moving_point_dict = {s:moving_coms[s] for s in brain_regions if s in common_keys}
        assert len(moving_point_dict) > 0, 'Not enough moving points.'

        R, t = umeyama(moving_points.T, allen_points.T)
        reg_points = R @ moving_points.T + t
        reg_point_dict = {s:reg_points.T[i] for i,s in enumerate(brain_regions) if s in common_keys}
        distances = []
        sortme = {}
        for structure in common_keys:
            (x,y,z) = allen_point_dict[structure]
            allen_point = np.array([x,y,z])    
            moving_point = np.array(moving_point_dict[structure])
            reg_point = brain_to_atlas_transform(moving_point, R, t)
            d = self.calculate_distance(allen_point, reg_point)
            distances.append(d)
            sortme[structure] = d

        ds = {k: v for k, v in sorted(sortme.items(), key=lambda item: item[1])}
        # 1st dataframe = distances
        df_distance = pd.DataFrame(ds.items(), columns=['Structure', 'distance'])
        csvfilename = os.path.join(self.csv_path, f'{csvfile}_distance.csv')
        df_distance.to_csv(csvfilename, index = False)

        # 2nd dataframe = allen
        df_allen = pd.DataFrame(allen_point_dict.items(), columns=['Structure', 'xyz'])
        df_allen['S'] = df_allen.apply (lambda row: self.label_left_right(row), axis=1)
        df_allen[['X', 'Y', 'Z']] = pd.DataFrame(df_allen['xyz'].tolist(), index=df_allen.index)
        csvfilename = os.path.join(self.csv_path, f'{csvfile}_allen.csv')
        df_allen.to_csv(os.path.join(csvfilename), index = False)

        # save 3rd dataframe = atlas
        df_atlas = pd.DataFrame(reg_point_dict.items(), columns=['Structure', 'xyz'])
        df_atlas['S'] = df_atlas.apply (lambda row: self.label_left_right(row), axis=1)
        df_atlas[['X', 'Y', 'Z']] = pd.DataFrame(df_atlas['xyz'].tolist(), index=df_atlas.index)
        csvfilename = os.path.join(self.csv_path, f'{csvfile}_atlas.csv')
        df_atlas.to_csv(csvfilename, index = False)


    def fetch_allen_origins(self):
        structures = {
            '3N_L': (354.00, 147.00, 216.00),
            '3N_R': (354.00, 147.00, 444.00),
            '4N_L': (381.00, 147.00, 214.00),
            '4N_R': (381.00, 147.00, 442.00),
            '5N_L': (393.00, 195.00, 153.00),
            '5N_R': (393.00, 195.00, 381.00),
            '6N_L': (425.00, 204.00, 204.00),
            '6N_R': (425.00, 204.00, 432.00),
            '7N_L': (415.00, 256.00, 153.00),
            '7N_R': (415.00, 256.00, 381.00),
            '7n_L': (407.00, 199.00, 157.00),
            '7n_R': (407.00, 199.00, 385.00),
            'AP': (495.00, 193.00, 217.00),
            'Amb_L': (454.00, 258.00, 167.00),
            'Amb_R': (454.00, 258.00, 395.00),
            'DC_L': (424.00, 177.00, 114.00),
            'DC_R': (424.00, 177.00, 342.00),
            'IC': (369.00, 44.00, 141.00),
            'LC_L': (424.00, 161.00, 185.00),
            'LC_R': (424.00, 161.00, 413.00),
            'LRt_L': (464.00, 262.00, 150.00),
            'LRt_R': (464.00, 262.00, 378.00),
            'PBG_L': (365.00, 141.00, 138.00),
            'PBG_R': (365.00, 141.00, 366.00),
            'Pn_L': (342.00, 139.00, 119.00),
            'Pn_R': (342.00, 139.00, 347.00),
            'RtTg': (353.00, 185.00, 161.00),
            'SC': (329.00, 41.00, 161.00),
            'SNC_L': (313.00, 182.00, 148.00),
            'SNC_R': (313.00, 182.00, 376.00),
            'SNR_L': (310.00, 175.00, 137.00),
            'SNR_R': (310.00, 175.00, 365.00),
            'Sp5C_L': (495.00, 202.00, 136.00),
            'Sp5I_L': (465.00, 202.00, 127.00),
            'Sp5I_R': (465.00, 202.00, 355.00),
            'Sp5O_L': (426.00, 207.00, 137.00),
            'Sp5O_R': (426.00, 207.00, 365.00),
            'VLL_L': (361.00, 149.00, 137.00),
            'VLL_R': (361.00, 149.00, 365.00),
        }
        return structures