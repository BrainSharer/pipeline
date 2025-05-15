"""
This takes the coordinates and packs them into a binary file,
see https://github.com/google/neuroglancer/issues/227
Create a dir on birdstore called points
put the info file under points/info
create the binary file and put in points/spatial0/0_0_0
"""
import argparse
import json
import os
import struct
import sys
import shutil
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import gzip


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.image_manipulation.image_manager import ImageManager
from library.image_manipulation.filelocation_manager import FileLocationManager, data_path
from library.controller.sql_controller import SqlController

def create_points(animal, scaling_factor=32, debug=False):
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    coordinates = []
    session_id = 8061    
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    w = sqlController.scan_run.width//scaling_factor
    h = sqlController.scan_run.height//scaling_factor
    z_length = 440

    polygons = sqlController.get_annotation_volume(session_id, scaling_factor=1)
    xyresolution = sqlController.scan_run.resolution
    zresolution = sqlController.scan_run.zresolution

    path = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(path):
        print(f'Removing existing directory {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    coordinates = []

    for section, points  in polygons.items():
        converted_points = [(p[0]/xyresolution/scaling_factor, p[1]/xyresolution/scaling_factor) for p in points]
        z = int(round(section / zresolution))
        if debug:
            print(f'{z=} {converted_points[0]=}')
        for point in converted_points:
            x, y = point
            coordinates.append((int(x), int(y), int(z)))

    coordinates = coordinates[:10]
    points = np.array(coordinates)
    shape = (w, h, z_length)


    print(f'length of coordinates: {len(coordinates)} {points.shape=} mean={np.mean(points, axis=0)} {shape=}')
    
    spatial_dir = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0', 'spatial0')
    info_dir = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(info_dir):
        print(f'Removing existing directory {info_dir}')
        shutil.rmtree(info_dir)
    os.makedirs(spatial_dir, exist_ok=True)
    point_filename = os.path.join(spatial_dir, '0_0_0.gz')
    info_filename = os.path.join(info_dir, 'info')
    
    with open(point_filename,'wb') as outfile:
        total_count=len(coordinates) # coordinates is a list of tuples (x,y,z) 
        buf = struct.pack('<Q',total_count)
        
        for (x,y,z) in coordinates:
            #print(x,y,z)
            pt_buf = struct.pack('<3f',x,y,z)
            buf+=pt_buf

        # write the ids at the end of the buffer as increasing integers 
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        buf+=id_buf
        bufout = gzip.compress(buf)
        outfile.write(bufout)
    

    #junk1 = "len coords = %s" % len(coordinates)
    #print(f'{junk1=}')
    #print(*range(len(coordinates)))
    #return

    info = {}
    spatial = {}
    properties = {}
    spatial["chunk_size"] = shape
    spatial["grid_shape"] = [1, 1, 1]
    spatial["key"] = "spatial0"
    spatial["limit"] = 10000
    properties["id"] = "color"
    properties["type"] = "rgb"
    properties["default"] = "red"
    properties["id"] = "size"
    properties["type"] = "float32"
    properties["default"] = "10"
    properties["id"] = "p_uint8"
    properties["type"] = "int8"
    properties["default"] = "10"

    info["@type"] = "neuroglancer_annotations_v1"
    info["annotation_type"] = "POINT"
    info["by_id"] = {"key":"by_id"}
    info["dimensions"] = {"x":[str(xyresolution/scaling_factor*1000),"um"],
                        "y":[str(xyresolution/scaling_factor*1000),"um"],
                        "z":[str(int(zresolution)),"um"]}
    info["lower_bound"] = [0,0,0]
    info["upper_bound"] = shape
    info["properties"] = [properties]
    info["relationships"] = []
    info["spatial"] = [spatial]    

    with open(info_filename, 'w') as infofile:
        json.dump(info, infofile, indent=2)
        print('Info')
        print(info)
        print(info_filename)


def create_cloud_volume(animal):
    session_id = 8061    
    scaling_factor = 32
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    polygons = sqlController.get_annotation_volume(session_id, scaling_factor=1)
    xyresolution = sqlController.scan_run.resolution
    zresolution = sqlController.scan_run.zresolution

    w = sqlController.scan_run.width//scaling_factor
    h = sqlController.scan_run.height//scaling_factor
    z_length = len(os.listdir(fileLocationManager.get_directory(channel=1, downsample=True, inpath='aligned')))
    path = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(path):
        print(f'Removing existing directory {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    shape = (w, h, z_length)  # Shape of your volume
    #coordinates = defaultdict(list)
    volume = np.zeros(shape, dtype=np.uint8)  # Initialize the volume with zeros
    for section, points  in polygons.items():
        contour_points = np.array([(p[0]/xyresolution/scaling_factor, p[1]/xyresolution/scaling_factor) for p in points]).astype(np.int32)

        z = int(round(section / zresolution))
        #coordinates[z].append(converted_points)
        volume_slice = np.zeros((w, h), dtype=np.uint8)  # Create a slice for the current z
        cv2.polylines(volume_slice, [contour_points], isClosed=True, color=255, thickness=10)
        volume[:,:,z] = volume_slice

    xy = (xyresolution*1000)/scaling_factor
    resolution =(xy, xy, int(zresolution*1000))
    
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='image',  # or 'segmentation' if you're using labels
        data_type=np.uint8,   # or 'uint32' for segmentation
        encoding='raw',
        resolution=resolution,
        voxel_offset=(0, 0, 0),
        chunk_size=(32,32,32),
        #volume_size=volume.shape[::-1],  # x,y,z
        volume_size=shape,  # x,y,z
    )
    tq = LocalTaskQueue(parallel=1)

    vol = CloudVolume(f'file://{path}', info=info)
    vol.commit_info()
    vol[:,:,:] = volume
    tasks = tc.create_downsampling_tasks(f'file://{path}', mip=0, num_mips=1, compress=True)
    tq.insert(tasks)
    tq.execute()




def create_pointsXXX(animal, layer, session_id, debug=False):

    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    # Get src points data
    arr = np.array([[1000,2000,300],[1100,2000,300],[1200,2000,300]])
    scan_run = sqlController.scan_run
    sections = sqlController.get_section_count(animal)
    width = scan_run.width
    height = scan_run.height
    xyresolution = scan_run.resolution
    zresolution = scan_run.zresolution
    scales = np.array([xyresolution, xyresolution, zresolution])
    arr = arr / scales

    df = pd.DataFrame(arr, columns = ['x','y','z'])
    df['Layer'] = [layer for l in range(arr.shape[0])]
    df.sort_values(by=['z','x','y'], inplace=True)
    records = df[['x', 'y', 'z']].to_records(index=False)
    coordinates = list(records)
    layer = str(layer).replace(' ','_')
    OUTPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, layer)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    infofile_path = os.path.join(data_path, 'atlas_data', 'points', 'info')
    outfile = os.path.join(OUTPUT_DIR, 'info')
    with open(infofile_path, 'r+') as rf:
        data = json.load(rf)
        data['upper_bound'] = [width, height, sections]  # <--- add `id` value.
        rf.seek(0)  # <--- should reset file position to the beginning.
    with open(outfile, 'w') as wf:
        json.dump(data, wf, indent=4)

    spatial_dir = os.path.join(OUTPUT_DIR, 'spatial0')
    os.makedirs(spatial_dir)
    total_count = len(coordinates)  # coordinates is a list of tuples (x,y,z)

    filename = os.path.join(spatial_dir, '0_0_0.gz')
    with open(filename, 'wb') as outfile:
        buf = struct.pack('<Q', total_count)
        pt_buf = b''.join(struct.pack('<3f', x, y, z) for (x, y, z) in coordinates)
def calculate_factors(level):
    """ 
    ---PURPOSE---
    Calculate the downsampling factor to apply to the grid_shape/chunk size at a given spatial index level.
    This is chosen to make the chunks as isotropic as possible, change as needed for your volume
    ---INPUT---
    level     - 0-indexed integer representing the spatial index level
    ---OUTPUT---
    d[level]  - The downsampling factor to apply to the level to get to the next level
    """
    # 
    d = {}
    d[0] = [1,1,1]
    d[1] = [2,2,1]
    d[2] = [2,2,1]
    for i in range(3,20):
        d[i] = [2,2,2]
    return d[level]

def make_cells(grid_shape):
    """ 
    ---PURPOSE---
    Make a list of grid cells e.g. ["0_0_0","1_0_0", ...] given a grid shape
    ---INPUT---
    grid_shape  - number of cells at a given level in each coordinate as a list,
                  e.g. [4,4,2] means 4x4x2 grid in x,y,z
    ---OUTPUT---
    cells       - A list of strings representing the cells, 
                  e.g. ['0_0_0', '0_1_0', '1_0_0', '1_1_0']
    """
    cells = []
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for z in range(grid_shape[2]):
                cell = f"{x}_{y}_{z}"
                cells.append(cell)
    return cells

def get_child_cells(cell,factor):
    """ 
    ---PURPOSE---
    Given a cell string e.g. 1_2_3 and a downsampling factor, e.g. [2,2,1]
    figure out all of the child cells of this cell in the next spatial index level 
    ---INPUT---
    grid_shape  - number of cells at a given level in each coordinate as a list,
                  e.g. [4,4,2] means 4x4x2 grid in x,y,z
    ---OUTPUT---
    cells       - A list of strings representing the cells, 
                  e.g. ['0_0_0', '0_1_0', '1_0_0', '1_1_0']
    """
   
    child_cells = []
    xcell,ycell,zcell = [int(x) for x in cell.split('_')] # n,m,p
    xfactor,yfactor,zfactor = factor # x,y,z
    for xf in range(0,xfactor):
        x_child = xcell*xfactor + xf
        for yf in range(0,yfactor):
            y_child = ycell*yfactor + yf
            for zf in range(0,zfactor):
                z_child = zcell*zfactor + zf
                child_cell = f"{x_child}_{y_child}_{z_child}"
                child_cells.append(child_cell)
    return child_cells

def save_cellfile(level,cell,coordinates,debug=False):
    """ 
    ---PURPOSE---
    Save the binary spatially indexed grid cell file,
    e.g. if level=1 and cell="1_1_0", then the file will be: spatial1/1_1_0 
    Assumes the global variable layer_dir is defined which is the 
    directory in which to save the spatial index directories
    ---INPUT---
    level       - 0-indexed integer representing the spatial index level
    cell        - a string like "0_0_0" representing the x,y,z grid location at a given level 
                  in which you want to extract a subset
    coordinates - a 2D array of coordinates like array([x0,y0,z0],...[xN,yN,zN])
    debug       - if True prints out that it saved the file
    ---OUTPUT---
    Writes the file, but does not return anything
    """
    # We already know how to encode just the coordinates. Do it like so for the first 100 points
    animal = "MD594"
    fileLocationManager = FileLocationManager(animal)
    print(f'type of coordinates: {type(coordinates)}')
    print(f'shape of coordinates: {coordinates.shape}')
    print(f'dtype: {coordinates.dtype}')

    layer_dir = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    spatial_dir = os.path.join(layer_dir,f"spatial{level}")
    if not os.path.exists(spatial_dir):
        os.mkdir(spatial_dir)
    filename = os.path.join(spatial_dir,cell)
    total_count = len(coordinates)
    with open(filename,'wb') as outfile:
        buf = struct.pack('<Q',total_count)
        pt_buf = b''.join(struct.pack('<3f',x,y,z) for (x,y,z) in coordinates)
        buf += pt_buf
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        buf += id_buf
        outfile.write(buf)
    if debug:
        print(f"wrote {filename}")
    
def find_intersecting_coordinates(coordinates,lower_bounds,upper_bounds):
    """ 
    ---PURPOSE---
    Find the subset of coordinates that fall within lower and upper bounds in x,y,z
    ---INPUT---
    coordinates  - a 2D array of coordinates like array([x0,y0,z0],...[xN,yN,zN])
    lower_bounds - a tuple or list of x,y,z lower bounds like [0,0,0]
    upper_bounds - a tuple or list of x,y,z upper bounds like [2160,2560,617]
    ---OUTPUT---
    coordinates[mask] - the subset of coordinates that fall 
                        within the lower and upper bounds
    """
    mask = (coordinates[:,0]>=lower_bounds[0]) & (coordinates[:,0]<upper_bounds[0]) & \
           (coordinates[:,1]>=lower_bounds[1]) & (coordinates[:,1]<upper_bounds[1]) & \
           (coordinates[:,2]>=lower_bounds[2]) & (coordinates[:,2]<upper_bounds[2])
    return coordinates[mask]





def create_points(unique_coordinates,layer_dir,grid_shape = [1,1,1],
         chunk_size=[2160,2560,687],dimensions_m=[5e-06,5e-06,1e-05],
         limit=10000,debug=False):
    """ 
    ---PURPOSE---
    Create the multiple spatial index levels and save out the cell files at each level.
    Also create, save and return the info file for this layer.
    ---INPUT---
    unique_coordinates - A 2D array of all coordinates representing your point annotations
                         that you want to spatially index. Duplicates should be removed already.
    layer_dir          - Base precomputed layer directory in which to save the info file
                         and spatial index directories
    grid_shape         - The grid shape of level 0. Typically this is [1,1,1].
    chunk_size         - The chunk size of level 0. If grid_shape = [1,1,1] then this is 
                         the dimensions of the entire volume, e.g. [2160,2560,617]
    dimensions_m       - The x,y,z dimensions in meters in a tuple or list
    limit              - The maximum number of annotations you wish to display 
                         in any cell at any level in Neuroglancer
    debug              - Set to True to print out various quantities to help with debugging
             
    ---OUTPUT---
    Writes out each spatialX/X_Y_Z spatial index file in layer_dir
    Writes out the info file in layer_dir
    info    - a dictionary containing the precomputed info JSON information
    """
    # Complete all of the info file except for the spatial part
    info = {}
    info['@type'] = "neuroglancer_annotations_v1"
    info['annotation_type'] = "POINT"
    info['by_id'] = {'key':'by_id'}
    info['dimensions'] = {'x':[str(dimensions_m[0]),'m'],
                          'y':[str(dimensions_m[1]),'m'],
                          'z':[str(dimensions_m[2]),'m']}
    info['lower_bound'] = [0,0,0]
    info['upper_bound'] = chunk_size
    info['properties'] = []
    info['relationships'] = []
    info['spatial'] = []
    # Create layer dir if it doesn't exist yet
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    # initialize some variables
    level=0
    cell="0_0_0"
    
    total_annotations = len(unique_coordinates)
    remaining_annotations = {} # will hold the arrays of coordinates in each cell at each level
    remaining_annotations[level] = {cell:unique_coordinates}

    maxCount = {} # will hold the maximum remaining annotations at each level
    
    # Iterate over levels until there are no more annotations to assign to child cells
    while True:
        if debug:
            print("##############")
            print(f"Level: {level}")
            print("##############")
        
        # Figure out maxCount to see if we are out of cells
        N_annotations_this_level = [len(x) for x in remaining_annotations[level].values()]
        maxCount[level] = max(N_annotations_this_level)
        if maxCount[level] == 0:
            print("Finished! Writing info file:")
            info_path = os.path.join(layer_dir,"info")
            print(info_path)
            with open(info_path,'w') as outfile:
                json.dump(info,outfile,indent=2)
            break
        # If we made it past there then we have cells left to assign
    
        # Use utility functions to figure out grid_shape and chunk_size for this level
        factor = calculate_factors(level)
        grid_shape = [a*b for a,b in zip(grid_shape,factor)]
        chunk_size = [a/b for a,b in zip(chunk_size,factor)]
        # Make the spatial dict for the info file
        spatial_dict_this_level = {
        'key':f'spatial{level}',
        'grid_shape':grid_shape,
        'chunk_size':chunk_size,
        'limit':limit
        }
        info['spatial'].append(spatial_dict_this_level)
        
        cells = make_cells(grid_shape)
            
        if debug:
            print(f"chunk_size={chunk_size}, maxCount = {maxCount[level]}")
            print("Have these cells:", cells)
        
        # Figure out the probability of extracting each annotation based on the limit
        if maxCount[level] > limit:
            prob = limit/maxCount[level]
        else:
            prob = 1
            
        # Loop over each cell at this level
        for cell in cells:
            if debug:
                print("In cell: ", cell)
            
            # Look up the remaining annotations in this cell, which was computed during the last iteration
            annotations_this_cell = remaining_annotations[level][cell]            
            N_annotations_this_cell = len(annotations_this_cell)
            if debug:
                print(f"started with {N_annotations_this_cell} annotations")
            
            # Need to know the child cells and the size of each so we can figure out the 
            # remaining counts in each
            next_factor = calculate_factors(level+1)
            child_cells = get_child_cells(cell,next_factor)
            next_chunk_size = [a/b for a,b in zip(chunk_size,next_factor)]

            # If we have annotations in this cell, then save the spatial index file for this level and cell
            # If not, don't save the file since it would be empty
            if N_annotations_this_cell != 0:
                # Figure out the subset of cells based on the probability calculated above
                N_subset = int(round(N_annotations_this_cell*prob))
                
                # figure out list of indices of the remaining array to grab 
                subset_indices = np.random.choice(range(N_annotations_this_cell),size=N_subset,replace=False)
                # Use these indices to get the subset of annotations
                subset_cells = np.take(annotations_this_cell,subset_indices,axis=0)
                
                if debug:
                    print(f"subsetted {len(subset_cells)} annotations")

                # save these cells to a spatial index file
                save_cellfile(level,cell,subset_cells,debug=debug)
                
                # Figure out the leftover annotations that weren't included in the subset
                indices_annotations_this_cell = range(len(annotations_this_cell))
                leftover_annotation_indices = set(indices_annotations_this_cell)-set(subset_indices)
                leftover_annotations = np.take(annotations_this_cell,list(leftover_annotation_indices),axis=0)
                if debug:
                    print(f"have {len(leftover_annotations)} annotations leftover")
            else:
                leftover_annotations = np.array([])
            # Initialize the next level in the remaining_annotations dictionary
            if level+1 not in remaining_annotations.keys():
                remaining_annotations[level+1] = {}
            
            if debug:
                print("Looping over child cells: ", child_cells)
            
            # Intiailize a variable to keep track of how many annotations total are in each child cell
            n_annotations_in_child_cells = 0
            
            # Loop over child cells and figure out how many of the remaining annotations 
            # fall in each child cell region
            for child_cell in child_cells:
                if N_annotations_this_cell == 0:
                    remaining_annotations[level+1][child_cell] = np.array([])
                    continue
                
                if debug:
                    print(f"Child cell: {child_cell}")
                
                # figure out which of the leftover annotations fall within this child cell
                child_cell_indices = [int(x) for x in child_cell.split('_')]
                child_lower_bounds = [a*b for a,b in zip(child_cell_indices,next_chunk_size)]
                child_upper_bounds = [a+b for a,b, in zip(child_lower_bounds,next_chunk_size)]
                
                if debug:
                    print("Child lower and upper bounds")
                    print(child_lower_bounds)
                    print(child_upper_bounds)

                # Now use the bounds to find intersecting annotations in this child cell
                intersecting_annotations_this_child = find_intersecting_coordinates(
                    leftover_annotations,child_lower_bounds,child_upper_bounds)
                
                if debug:
                    print(f"Have {len(intersecting_annotations_this_child)} in this child cell")
                
                # Assign the remaining annotations for the child cell in the dictionary
                remaining_annotations[level+1][child_cell] = intersecting_annotations_this_child
                
                n_annotations_in_child_cells+=len(intersecting_annotations_this_child)
            
            # Make sure that the sum of all annotations in all child cells equals the total for this cell
            if debug:
                print("Leftover annotations this cell vs. sum in child cells")
                print(len(leftover_annotations),n_annotations_in_child_cells)
        assert len(leftover_annotations) == n_annotations_in_child_cells
        
        # increment to the next level before next iteration in while loop
        level+=1
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--session_id', help='Session ID', required=False, default=1)
    parser.add_argument('--scaling_factor', help='scaling factor', required=True, default=32)
    parser.add_argument('--layer', help='layer', required=False, default='test_layer')
    parser.add_argument('--debug', help='print info', required=False, default='true')

    args = parser.parse_args()
    animal = args.animal
    layer = args.layer
    session_id = int(args.session_id)
    scaling_factor = int(args.scaling_factor)
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    #create_points(animal, scaling_factor, debug)
    create_cloud_volume(animal)

    session_id = 8061    
    scaling_factor = 1
    fileLocationManager = FileLocationManager(animal)
    thumbnail_aligned = fileLocationManager.get_directory(channel=1, downsample=True, inpath='aligned')
    image_manager = ImageManager(thumbnail_aligned)
    sqlController = SqlController(animal)
    polygons = sqlController.get_annotation_array(session_id)
    xyresolution = sqlController.scan_run.resolution
    zresolution = sqlController.scan_run.zresolution

    w = sqlController.scan_run.width//scaling_factor
    h = sqlController.scan_run.height//scaling_factor
    path = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(path):
        print(f'Removing existing directory {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    shape = (w, h, image_manager.len_files)  # Shape of your volume

    unique_coordinates = np.array([
        [ 495, 1289,   47],
        [ 489, 1473,   45],
        [ 483, 1453,   49],
        [ 467, 1409,   50],
        [ 497, 1490,   46],
        [ 504, 1293,   45],
        [ 459, 1398,   50],
        [ 504, 1297,   45],
        [ 462, 1412,   49],
        [ 478, 1310,   50],
        [ 505, 1421,   46],
        [ 501, 1244,   50],
        [ 506, 1243,   50],
        [ 482, 1298,   50]]
    )

    shape = [2160,2560,687]
    info = create_points(unique_coordinates, path, grid_shape = [1,1,1],
            chunk_size=shape,limit=10000,debug=False)    
    #create_cloud_volume(animal)
    print(info)

