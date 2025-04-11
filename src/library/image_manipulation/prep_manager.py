import os, re
import json
import glob
import paramiko
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from library.utilities.utilities_process import get_image_size, test_dir


try:
    from settings import host, password, user, schema
    ssh_password = str(password)
    ssh_user = str(user)
    remote_host = 'imageserv.dk.ucsd.edu'
except ImportError as fe:
    print('You must have a settings file in the pipeline directory.', fe)
    raise

class PrepCreater:
    """Contains methods related to generating low-resolution images from image stack 
    [so user can review for abnormalities
    e.g. missing tissue, poor scan, etc.] and applying this quality control analysis 
    to image stack
    """

    def apply_QC(self):
        """Applies the inclusion and replacement results defined by the user on the Django admin portal for the Quality Control step
        to the full resolution images.  The result is stored in the animal_folder/preps/CHX/full directory
        Note: We don't want the image size when we are downsampling, only at full resolution.
        """

        if self.downsample:
            self.input = self.fileLocationManager.thumbnail_original
            self.output = self.fileLocationManager.get_thumbnail(self.channel)
        else:
            self.input = self.fileLocationManager.tif
            self.output = self.fileLocationManager.get_full(self.channel)
        
        if not os.path.exists(self.input):
            """This checks for the thumbnail_original dir. This might not be available with the original brains
            The data will then be in the regular thumbnail dir
            """
            print(f'This dir does not exist. {self.input}')
            print(f'Checking the regular thumbnail dir')
            self.input = self.output
            if not os.path.exists(self.input):
                return

        try:
            starting_files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return
            
        self.fileLogger.logevent(f"Input FOLDER: {self.input}")
        self.fileLogger.logevent(f"INPUT FOLDER FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {self.output}")
        os.makedirs(self.output, exist_ok=True)
        try:
            sections = self.sqlController.get_sections(self.animal, self.channel, self.debug)
        except:
            raise Exception('Could not get sections from database')
        
        self.fileLogger.logevent(f"DB SECTIONS [EXPECTED OUTPUT FOLDER FILE COUNT]: {len(sections)}")

        for section_number, section in enumerate(sections):
            infile = os.path.basename(section.file_name)
            input_path = os.path.join(self.input, infile)
            output_path = os.path.join(self.output, str(section_number).zfill(3) + ".tif")
            
            if not os.path.exists(input_path):
                print(f"MISSING SRC FILE: {section_number=}; {input_path}: SKIPPING SYMBOLIC LINK (CHECK DB OR RENAME FILE)")
                continue

            if os.path.exists(output_path):
                continue

            if not self.downsample:
                width, height = get_image_size(input_path)
                self.sqlController.update_tif(section.id, width, height)

            if self.debug:
                print(f'Creating symlink to {output_path}')

            try:    
                relative_input_path = os.path.relpath(input_path, os.path.dirname(output_path))
                os.symlink(relative_input_path, output_path)
            except Exception as e:
                print(f"CANNOT CREATE SYMBOLIC LINK: {output_path} {e}")
                

    def create_symbolic_link(self, target_path: str, link_path: str):
        if self.debug:
            print(f"Creating symbolic link from {target_path} to {link_path}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(remote_host, username=ssh_user, password=ssh_password)

            # Check if the symlink already exists
            check_command = f"if [ -L {link_path} ]; then echo 'exists'; else echo 'not_exists'; fi"
            stdin, stdout, stderr = ssh.exec_command(check_command)
            result = stdout.read().decode().strip()

            if result != "exists":
                command = f"ln -s {target_path} {link_path}"
                stdin, stdout, stderr = ssh.exec_command(command)

                if self.debug:
                    print("Output:", stdout.read().decode())
                    print("Error:", stderr.read().decode())
            else:
                print(f"Symbolic link {link_path} already exists. Skipping creation.")

        except Exception as e:
            if self.debug:
                print(f"An error occurred: {e}")
        finally:
            ssh.close()


    def gen_ng_preview(self):
        # GET CHANNEL NAMES FROM meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
        with open(meta_store, 'r') as f:
            data = json.load(f)
        tracing_data = data.get("Neuroanatomical_tracing", {})
        channel_names = [info.get("channel_name") for info in tracing_data.values()]

        #IF AVAIABLE, GET XY RESOLUTION FROM meta-data.json, ELSE FROM DATABASE
        #TODO: remove hard-coded z resolution
        xy_resolution_unit = data.get("xy_resolution_unit", {})
        x_resolution = float(xy_resolution_unit[0])
        y_resolution = float(xy_resolution_unit[2])
        z_resolution = 0.00002  # hard-coded z resolution
        dimensions = {
            "x": [x_resolution, "m"],
            "y": [y_resolution, "m"],
            "z": [z_resolution, "m"]
        }

        # CHECK IF NG-COMPATIBLE IMAGE LAYERS WERE GENERATED
        pattern1 = re.compile(r'.*_(re)?aligned$')
        pattern2 = re.compile(r'.*\.zarr$')
        pattern3 = re.compile(r'^C\d+$')  # starts with 'C' and then only digits
        pattern_rechunkme = re.compile(r'rechunkme')   

        ng_folders = [
            f for f in glob.glob(os.path.join(self.fileLocationManager.neuroglancer_data, '*'))
            if os.path.isdir(f) and
                not pattern_rechunkme.search(os.path.basename(f)) and 
                (pattern1.match(os.path.basename(f)) or
                pattern2.match(os.path.basename(f)) or
                (pattern3.match(os.path.basename(f)) and not pattern1.match(os.path.basename(f)))
            )
        ]
        
        img_layers = {}
        for channel_name in channel_names:
            ome_zarr_path = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + ".zarr")
            if self.downsample:
                precomputed_path1 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + 'T')
                precomputed_path2 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + 'T_aligned')
            else:    
                precomputed_path1 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name)
                precomputed_path2 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + '_aligned')
            if ome_zarr_path in ng_folders:
                folder_name = os.path.basename(ome_zarr_path)
                print(f"FOR {channel_name}, USING OME-ZARR DIR: {ome_zarr_path}")
                img_layers[channel_name] = {'src': ome_zarr_path, 'src_type': 'zarr2', 'folder_name': folder_name}
            elif precomputed_path1 in ng_folders:
                img_layers[channel_name] = {'src': precomputed_path1, 'src_type': 'precomputed', 'folder_name': channel_name}
                print(f"USING PRECOMPUTED DIR: {precomputed_path1}")
            elif precomputed_path2 in ng_folders:
                img_layers[channel_name] = {'src': precomputed_path2, 'src_type': 'precomputed', 'folder_name': channel_name}
                print(f"USING PRECOMPUTED DIR: {precomputed_path2}")
            else:
                print(f"ERROR: NEUROGLANCER DATA NOT FOUND FOR {channel_name}")
                continue

        ##################################################################################
        #define initial view field (x,y,z) - center on image and across stack
        if self.downsample:
            check_dir = self.fileLocationManager.get_thumbnail()
        else:
            check_dir = self.fileLocationManager.get_full()
        _, nfiles, max_width, max_height = test_dir(self.animal, check_dir, section_count=0)
        view_field = {"position": [max_width//2, max_height//2, nfiles//2]}

        ##################################################################################

        #TODO: CREATE WITH LAYERS WE HAVE
        # RAM: ?
        # ann_layer: name, rendering
        # segmentation: name, rendering

        #not sure what this is (but was in template)
        other_stuff = {"crossSectionScale": 121.51041751873494,
                "crossSectionDepth": 46.15384615384615,
                "projectionScale": 131072}

        ##################################################################################
        #COMPILE IMG SRC FOR LAYERS
        base_url = f"https://imageserv.dk.ucsd.edu/data/{self.animal}/neuroglancer_data/"
        
        ng_layers = []
        for channel_name, channel_attributes in img_layers.items():
            layer = {
                "type": "image",
                "source": f"{channel_attributes['src_type']}://{base_url}{channel_attributes['folder_name']}",
                "tab": "rendering",
                "name": channel_name
            }
            ng_layers.append(layer)

        dimensions_json = {"dimensions": dimensions}
        layers_json = {"layers": ng_layers}

        ##################################################################################
        #add annotation layers
        #TODO: dimensions from primary dimensions

        #see cell_manager.py (extract_predictions) for reverse engineering  [csv -> annotation format]

        annotations_json = {
            "type": "annotation",
            "source": {
                "url": "local://annotations",
                "transform": {
                "outputDimensions": {
                    "x": [
                    x_resolution,
                    "m"
                    ],
                    "y": [
                    y_resolution,
                    "m"
                    ],
                    "z": [
                    z_resolution,
                    "m"
                    ]
                }
                }
            },
            "tab": "source",
            "annotations": [],
            "annotationProperties": [
                {
                "id": "color",
                "description": "color",
                "type": "rgb",
                "default": "#ffff00"
                },
                {
                "id": "visibility",
                "description": "visibility",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 1,
                "step": 1
                },
                {
                "id": "opacity",
                "description": "opacity",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 1,
                "step": 0.01
                },
                {
                "id": "point_size",
                "description": "point marker size",
                "type": "float32",
                "default": 5,
                "min": 0,
                "max": 10,
                "step": 0.01
                },
                {
                "id": "point_border_width",
                "description": "point marker border width",
                "type": "float32",
                "default": 3,
                "min": 0,
                "max": 5,
                "step": 0.01
                },
                {
                "id": "line_width",
                "description": "line width",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 5,
                "step": 0.01
                }
            ],
            "name": "annotation"
            }


        #general settings **TODO: needs edit
        gen_settings = {"selectedLayer": {
                "visible": True,
                "layer": "C2"
            },
            "layout": "4panel",
            "helpPanel": {
                "row": 2
            },
            "settingsPanel": {
                "row": 3
            },
            "userSidePanel": {
                "tab": "User",
                "location": {
                "row": 1
                }
            }}

        combined_json = {**dimensions_json, **view_field, **other_stuff, **layers_json, **annotations_json, **gen_settings}
        print(json.dumps(combined_json, indent=2))


        #save to DB:
        '''
        id (auto)
        neuroglancer_state
        FK_lab_id
        created
        updated
        user_data
        comments
        description
        FK_user_id
        readonly
        public
        active
        '''

                

        #TODO: add more targeted link to only expose channels of interest on imageserv
        target_path = str(self.fileLocationManager.www)
        link_path = str(Path('/', 'srv', self.animal))
        self.create_symbolic_link(target_path, link_path)