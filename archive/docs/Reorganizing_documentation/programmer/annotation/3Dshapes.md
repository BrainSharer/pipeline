### Interaction design and tables for 3D shapes

3D shapes are expressed in two modalities

1. Contour-sequence: a sequence of 2D polygons, one per section, in
stack coordinates. This modality supports drawing contours on
sections, and creating/updating contours.

2. 3d blob: a representation of the 3D shape of the structure in
atlas coordinates. For example a 3D blob can be a particular 
landmark in the atlas.currently stored as a voxel map in a numpy
array. The 3D blob can be exported as a 3D mesh, or as
a contour sequence to be displayed on the sections.

#### Annotator tools

Neuroglancer should support creating and altering contour
sequences. Creating is by clicking consective points along the
perimeter (contours should be connected). Altering includes moving an
existing point, deleting a point and adding a point.

To aid the annotator and improve the continuity in the z direction,
one can copy contour from the previous section and make corrections.

#### Workflow

A 3D shape is initially created by the anatomist as a contour
sequence. One or more contour sequences for the same biological entity
are then combined into a 3D shape. Multiple contour sequences are used
to find the concensus of annotations from several anatomists/several brains (as we
have done for structures in the foundation brains). This operation
involves mappinng the annotations into atlas coordinates, finding the
concensus, smoothing, and generating a 3D blob (using single
voxels as the unit for large structures consumes memory and time, we
can do better by combining small and large cubes to make the shape).

The 3D blobs, together with the points (coms) are  some of the most valuable
data to be shared using BrainSharer. Version control here is critical
as over time people would add/alter shapes.

Given a stack and the coordinate transformation between the stack and
the atlas, the 3D blobs are transformed to stack coordinates and
sliced to create the contours to be drawn on the sections. 

Another transformation takes the 3D structure and makes it into a mesh
that can be viewed in neuroglancer in 3D.

[This approach transforms the atlas to the stack, rather than transforming the stack to the atlas, which is currently used and causes "striping" of the images.]

### Suggested New Tables

##### Contour sequence table
1. Contour Sequence ID
1. prep_id
1. Transformation  
1. generator (user_ID or ID of 3d-blob)  
1. name of 3D shape (Should probably have a table for these names, to
ensure consistency)  
1. First section number  
1. Last section number   

##### Contour table
1. ID,
1. Contour sequence ID
1. Section index
1. polygon (blob, pickle)

##### 3D blob

1. ID
1. Shape name (pointer to a table of shape names, including structures
1. and other things)
1. 3D shape (blob: pickle)
