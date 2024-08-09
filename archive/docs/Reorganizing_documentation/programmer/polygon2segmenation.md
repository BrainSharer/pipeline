## Converting polygons to segmentation layers

This document will describe the process of converting polygons that are drawn by an anatomist onto a Neuroglancer layer, into a separate 3D segmentation layer for Neuroglancer. The process described below starts after the anatomist has finished drawing all polygons and has saved this data into Neuroglancer.  

1. Information regarding the database polygon data saved by the anatomist:
    1. Data is stored in the `polygon_sequences` table.
    1. Each row contains an x,y,z coordinate (as floats) and has metadata describing the data point.  
    `yoav` I don't think there is a reason to store x,y,z in float. float does not imply more precision, just the ability to represent both very small or very large numbers. If we use 16 bit ints and set the unit distance to be a micrometer, we can represent offsets up to 65.536 mm. When computing distances between points or offsets, we should first convert to float64 and then, when storing, back to int16.  
    1. All offsets are measured in micrometers. micrometer.
    

`Yoav` I don't follow the description below. A class is not just a wrapper around a function, and instance of a class has state, i.e. variables that are maintained throughout the life of the instance. A description of a class should start with "this class defined an object that represents XXXX" and then be followed by the public methods and public variables. (the API defined by the class).   

1. Fetching the data involves importing the abakit module. This module uses the process described below:  

    The [PolygonSequenceController](https://github.com/ActiveBrainAtlas2/abakit/blob/dev/src/abakit/lib/Controllers/PolygonSequenceController.py) class querys the polygon sequence table by calling the `get_volume(prep_id,annotator_id,structure_id)` function.  The [PolygonSequenceController](https://github.com/ActiveBrainAtlas2/abakit/blob/dev/src/abakit/lib/Controllers/PolygonSequenceController.py) class initiates a sqalchemy session through the constructor of it's parent class [Controller](https://github.com/ActiveBrainAtlas2/abakit/blob/dev/src/abakit/lib/Controllers/Controller.py).

    The [Controller](https://github.com/ActiveBrainAtlas2/abakit/blob/dev/src/abakit/lib/Controllers/Controller.py) class contains methods to start the sql session with the right credentials, and the ability to add, delete, insert, and checking if a row exists.
    
`yoav` If I understand correctly, `controller` and it's subclasses are responsible for the connection with the database through sqlalchemy. The API should not depend on the fact that sqlalchemy is used. Initiating a session, with all of the associated parameters by importing a module called `session.py` is non-standard. Parameters should be read from a parameter file using a standard such as yaml. I would replace the name `controller` with something like `database_bridge` which better represents its function.

The `get_volume(prep_id,annotator_id,structure_id)` function queries the `annotation_session` table to find any active session matching the prep_id, annotator_id, structure_id. If a session is found, it queries the PolygonSequence table for the polygon points, and parses them into dictionaries.  Finally, the function returns the dictionary of polygon points for a specific structure.

1. Ater the data has been fetched, we can transform the polygon data to atlas space with the following process:
   
    The [TransformationController](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/lib/Controllers/TransformationController.py) class pulls the transformation stored in the transformation table using the `get_transformation` function.  The TransformationController class inherits from the same parent class Controller.
    
    We transform the points from the stack space to the atlas space using the `forward_transform_points` function. This function: `forward_transform_points` takes a list of point coordinates and outputs the result of the points after the transformation. The function will be run on all of the points in all the polygons and apply the affine or rigid transformation calculated from the structure coms to the atlas coms.

1. After the transformation, we create the 3D masks from polygon data
 
     The [VolumeMaker](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/atlas/VolumeMaker.py) class is then used to create a 3D mask of type uint8 from the 2D contours.  The aligned contours are given to the class by the function `set_aligned_contours`.  The volume is calculated by running the function `compute_COMs_origins_and_volumes`.  After the calculation, the volume can be found in the `self.volumes` attribute.

     The [VolumeMaker](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/atlas/VolumeMaker.py) class has the parent class [BrainStructureManager](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/atlas/BrainStructureManager.py) which inherits from the [Brain](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/lib/Brain.py) class and [VolumeUtilities](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/atlas/VolumeUtilities.py) class.  The Brain class has a multitude of functions that handle the file path and database access. VolumeUtilities has functions to threshold volumes and for smoothing the 3D mask with a gaussian filter.  Most of the function of the Brain class is not that useful to the VolumeMaker in this context.  We should add the ability to toggle these part of the functions so that a connection to the database would not be required to run this step.

1. Storing the 3D mask in the database

     Once the 3D mask has been created, it will then be stored in the `brain_shape` table. Each row in this table will hold metadata describing the row and also two blob columns. One blob column will hold the transformation. This transformation is pickled using the pickle module. The 2nd blob column will hold the pickled 3D mask. The largest of the structures (Superior Colliculus) is around 3MB. Saving the data to a pickle format uses this method: `np.ndarray.dumps(arr)`, where `arr` is the 3D mask.

1. Creating the Neuroglancer segmentation layer from the 3D masks
    1. This part of the process fetches the pickled data from the DB. The method to get the data from database blob to numpy array is: `arr = pickle.loads(brain_shape.numpy_data)`, where 'arr' is the numpy array we will use, and 'brain_shape.numpy_data' is the database blob column. Since each blob is a numpy array of zeros and ones, we then multiply each numpy array by a 'color' number taken from the database to give it a different color in Nueroglancer. This 3D mask is filled with mostly zeros and some color number. This array is then processed with the [Seung Lab Cloudvolume software](https://github.com/seung-lab/igneous). To process the numpy arrays into the segmentation layer, the following steps are taken:
        1. Fetches metadata from the DB to determine what shape the entire 3D volume will be. The width and height are taken from the width and height of the full scale images. These values are then downsampled by 32 (this is our universal downsampling factor). The numpy array would be too large with the full scale resolution. The z shape is taken from the number of sections.
        1. All structures (3D masks and x,y,z offsets) for that particular brain for that brain are fetched from the DB. The offsets are the distance in um taken from the top left origin of the Neuroglancer view.
        1. Each structure is then placed in the entire 3D volume we created with the width, height, section number data fetched earlier. Now we have a 3D volume with all the structures and this numpy array is then passed onto the Cloudvolume library to process into the segmentation layer.
        1. We can also create a single structure within the 3D volume space.
        1. The Cloudvolume process creates the precomputed data directory which can then be placed on Birdstore and can be accessed by the web server.
        
    Below is a description of the Seung Lab Cloudvolume module implementation:
    
1. The Cloudvolume module takes:
    1. A 3D mask of integers.
    1. A dictionary that contains the color and text related to each integer.
    1. Units and origins (information regarding the scales and offsets).
1. The code implementation is as follows:

   The [NgConverter](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/atlas/NgSegmentMaker.py) class converts the 3D mask to the segmentation layer.  The segmentation layer consists of a set of folders living on the file system. This class inherits from the [NumpyToNeuroglancer](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/lib/utilities_cvat_neuroglancer.py) class.  The [NumpyToNeuroglancer](https://github.com/ActiveBrainAtlas2/abakit/blob/master/src/abakit/lib/utilities_cvat_neuroglancer.py) was produced by Litao and is able to produce both Image layers and Segmentation layers from the 3D masks.

   The NumpyToNeuroglancer class uses the function `init_precomputed` to initiate the creation of a neuroglancer layer using the Seung lab CloudVolume package.  The specific calls to the package looks like:
   ```
   info = CloudVolume.create_new_info(
            num_channels=self.num_channels,
            layer_type=self.layer_type,  # 'image' or 'segmentation'
            data_type=self.data_type,  #
            encoding='raw',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
            resolution=self.scales,  # Size of X,Y,Z pixels in nanometers,
            voxel_offset=self.offset,  # values X,Y,Z values in voxels
            chunk_size=self.chunk_size,  # rechunk of image X,Y,Z in voxels
            volume_size=volume_size,  # X,Y,Z size in voxels
        )
    self.precomputed_vol = CloudVolume(f'file://{path}', mip=0, info=info, compress=True, progress=False)
    self.precomputed_vol.commit_info()
    self.precomputed_vol.commit_provenance()
   ```
   Where the `path` variable is the output path.  Then the information of each segment(in this case brain regions), including the color value and text descriptions are set using the `add_segment_properties` function.

   The volume is passed to NgConverter by 

   ```self.precomputed_vol[:, :, :] = self.volume```

   And the segmentation layer is created by 
   ```
    self.add_downsampled_volumes()
    self.add_segmentation_mesh()
   ```

1. The Cloudvolume module creates:
    
    1. The segmentation layer viewable in Neuroglancer. This is a directory containing a JSON file which has the scales, chunk sizes and directory locations that Neuroglancer needs to display the data. This directory is on Birdstore and is then accessible by our web server.

