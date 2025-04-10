How to make an Atlas
--------------------

This document will describe the steps used to create the most recent version of the DK atlas.

The data used to create this atlas was initially made by several anatomists working at the DK lab at UCSD.
Structures were drawn onto full resolution mouse brain images and the polygon vertices were then stored as CSV files
on the network filesystem. Three different mouse brains were used and they are collectively known as the foundation brains
and they are listed below:

- MD585
- MD589
- MD594

MD589 is used as the reference brain. There are 51 brain structures that were drawn by the anatomists. Not every
anatomist drew every structure on each of the three brains. This results in a varying number of drawn polygons per structure per brain.
Each brain has it's own CSV file and the process parses each CSV file. The first 10 rows of MD589 are show below:

.. code-block:: XML

	name	creator	orientation	section	side	time_created	vertices
	VCA	yuncong	sagittal	367	R	9152016033619	[[15450.64315379 13041.15143737]\n [15292.3766...
	IO	yuncong	sagittal	260	R	9122016211223	[[25199.46145603 17916.30221869]\n [24976.1090...
	LRt	yuncong	sagittal	286	R	9152016024654	[[29069.49187508 17203.67921491]\n [28909.8937...
	LRt	yuncong	sagittal	276	R	9152016024654	[[29219.12776439 16422.80664145]\n [29084.8059...
	SC	yuncong	sagittal	202	S	5312018021355	[[21584.44352233 7847.78193722]\n [21497.5691...

Looking at the data above, we can see that for each section of each structure, multiple anatomists may have drawn polygons.
These polygons are averaged together on each section and then these sections are concatenated to form a volume.  
For each structure, the minimum x,y, and z values are found.  These values are used to create a bounding box with the upper left corner
of the box being the origin. The origin is saved and the mean of all the origins is subtracted from each origin. This is 
where the bounding box is placed within the atlas.
For each brain section, the set of points is drawn on this bounding box. All the sections are then appended to form the box and
within the box is the structure. The box is saved as a numpy array with either zeroes or the value 255. 
Centers of mass are calculated for each of the drawn volumes
and these are then used to create a rigid transformation matrix that is used to align the drawn polygons to the reference brain.
This process now yields the origin of the bounding box, the actual 3D volume and then the center of mass. All this data is initially
stored on the filesystem and the centers of mass are also stored in the database.

In the following example, the SC structure has 3 drawn volumes that now need to be merged. They are first aligned
to the largest of the 3 volumes using a rigid transformation. 

Here is a preview of what the structure superior colliculus (SC) looks like at the mid z value:

.. image:: /_static/SC.unmerged.png

Now that they are aligned, the following steps are peformed to get a single smooth volume:

- The three volumes are smoothed together with a gaussian filter with a large standard deviation.
- This results in a volume of floats. An abitrary value of 150 is used where the values above 150 are set to the Allen ID and all other \
  values are set to zero. 
- The 3D volume is now set to be a binary volume with values of 0 and the Allen ID. Since the Allen IDs are large integers, the volume is saved as a 32bit Integer.

The structure superior colliculs (SC) looks like this after using smoothing and merging:

.. image:: /_static/SC.merged.png


The same process is repeated for the remaining 50 structures.

Structures not in the foundation brains can also be merged into the atlas. These are created from polygons drawn in neuroglancer
by the anatomists. The polygons are stored in the database, retreived and the same process as the foundation brain structures is repeated.
One problem is getting a rigid transformation between the neurotrace brains and the foundation brains. An Elastix rigid transformation
is calculated using the full 3D volume of a 10um isotropic MD589 brain and the neurotrace brain. This transformation is then applied to the neurotrace
data points which puts them into the same coordinate system as MD589.

We now have all the merged brain structures in MD589 10um isotropic space. The next step is to register this atlas with the Allen reference atlas.
We use the Allen SDK (https://allensdk.readthedocs.io/en/latest/) to get the centers of mass of common structures between the DK atlas and the Allen reference atlas.
One problem with this approach is structures have 
different names in the Allen SDK and the DK atlas. Another problem is the Allen atlas breaks some structures into sub structures (e.g. the SC structure).
Another issue is the DK atlas differentiates between left and right structures while the Allen does not.

A rigid transformation is calculated between the DK atlas and the Allen reference atlas using the centers of mass of the common structures.
Results are show below and all values are in micrometers:

.. csv-table:: Table Title
   :file: /_static/results.csv
   :header-rows: 1


With the rigid transformation, we can now create the 10um isotropic atlas. The 10um Allen atlas is of the following size:

- x_length = 1320
- y_length = 800
- z_length = 1140

We extend the x and y axes to accomodate the brain stem:

- x_length = 1820
- y_length = 1000
- z_length = 1140

The origin of each structure is retrieved and the new bounding box is calculated from the sizes listed above. This origin
is then transformed by the rigid transformation matrix between the DK atlas and the Allen reference atlas. Each
structure is then placed within the super volume and that super volume is then used to create a Neuroglancer dataset.

The atlas can be view in Neuroglancer at the following URL:

https://brainsharer.org/ng/?id=984

