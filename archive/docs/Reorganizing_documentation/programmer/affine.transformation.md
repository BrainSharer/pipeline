### Atlas coordinates
1. Structures from 3 reference brains (MD589, MD594, and MD585) are outlined by anatomists.
1. The coordinates from those structures are used to created a set of averaged structures which is the Atlas.
1. Yuncong used a virtual 3D space with units in 10microns. For example, the origin Yuncong derived for structure 3N_L is: [-165.58293065 -127.18518508  -38.94194041] 
1. That origin gets transformed into the COM in the Atlas box space with:
<pre>
 center = (origin + ndimage.measurements.center_of_mass(volume))
 com = atlas_box_center + center * atlas_raw_scale / atlas_box_scales
</pre>
1. Where:
    1. atlas_box_size=(1000, 1000, 300) # Arbitrary size of the box we are using
    1. atlas_box_scales=(10, 10, 20) # scales for Neuroglancer of the box
    1. atlas_raw_scale=10 # raw Yuncong scale of 10microns
    1. atlas_box_center = atlas_box_size / 2 # The center of our "Atlas Box"
1. resulting in: [-135.44472549 -108.49438786  -13.52838526] and this is placed in the database as the center of mass for the Atlas structure 3N_L
1. All the remaining structure COMs are computed with the same procedure.

### Goal
* make atlas coordinate units be microns
   * Longer term: make coordinates conform to the Allen Atlas.
* make brain coordinate units be micrones.
* Process: origin -> center -> com, com is the point that is saved in the database.
* Fixing the database: multiply x and y by 10, multiply z by 20
* Fixing Neuroglancer: translate from the transformed coordinates, which are in microns, to section/pixel coordinates.

### Coordinate systems:
* atlas coordinate system: in microns from center of brain.
* stack coordinate system: in microns from top left, first section
* Neuroglancer coordinate system: in pixels within a section, section number.

### Brain coordinates
1. Brain stack COMs are derived from Neuroglancer. For NTB brains, the resolution is 0.325 in the x and y directions and 20um in the z direction.
1. We get the Atlas and the brain coordinates into 1um space to make a common reference.
    1. Use atlas_box_scales = [10, 10, 20]
    1. Use brain reference_scales of [0.325, 0.325, 20]

1. With python we do:
<pre>
        common_keys = atlas_centers.keys() & reference_centers.keys() # get common structures
        atlas_point_set = np.array([atlas_centers[s] for s in structures if s in common_keys]).T
        atlas_point_set = np.diag(atlas_box_scales) @ atlas_point_set
        brain_point_set = np.array([reference_centers[s] for s in structures if s in common_keys]).T
        brain_point_set = np.diag(reference_scales) @ brain_point_set
</pre>
1. Then we send those two sets to the align_atlas method
