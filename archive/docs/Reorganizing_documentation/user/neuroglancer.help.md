## Neuroglancer help for end users
### Help topics are arranged by location on the window, starting with the top left and moving clockwise around the screen.
#### Top left
   
##### x,y,z coordinates
1. Cycling through x,y,z coordinates with the cursor. The x,y,z (section) coordinates are displayed in the top left of the screen.
You can place your cursor within the white area of each number and scroll through the min to max numbers. If there
   is only one layer, the vertical strip representing the min and max coordinates will be a solitary strip. Adding
   more layers will give the strip additional levels which might be of different min and maxes, depending on the layers.
   The scale measurement is also displayed next to each x,y,z coordinate.

##### Layers
1. Layers can define a view for channels, an atlas, shell or annotation layers. To adjust a setting of a view, you need to first select it. This is accomplished by using the mouse and clicking inside one of the rectangular boxes near the top left of the screen. If you click on the 'X' within the box, the layer will be hidden. To unhide the layer, simply reclick the 'X'. 
1. To completely remove a layer, do control click within the rectangular box. Once you do this, the layer is gone and if you want it back, 
you'll need to add the layer back.
1. To create a new layer for an image channel, or for the atlas, or for the shell, follow these instructions:
    1. Image channel - Click the plus sign next to the right of the rectangular boxes.
    1. This will show the source matrix on the right. 
    1. Look for the 'precomputed://' text box
    1. In that text box type: `https://activebrainatlas.ucsd.edu/data/DKXX/neuroglancer_data/C1`
    1. Subsitutue 'DKXX' for the brain you want and also substitute 'C1' for the channel you want, e.g., C1 or C2 or C3
    1. With your cursor at the end of that data url you just typed above, hit enter twice. This will pull up that channel. If you don't see anything, it may be because the layer is too dark or it is out of zoom.
1. To create a new annotation layer, follow these instruction: 
    1. Hold control down and click on the plus sign. This creates a local annotation layer where you can add points, ellipsoids, lines or boxes.
    1. Give the layer a meaningful name by clicking in the text box where it says 'annotation'. Type in a name and hit enter.
    1. To start adding points, the 'Annotation' tab should be highlighted on the right side of the page.
    1. Choose a color from the yellow color selector box.
    1. Choose an annotation type, point, bounding box, ellipsoid.
    1. Once you have selected a color and an annotation type, just hold the control key and click anywhere on one of the quadrants
    1. The x,x,z coordinates will show up in the box on the right side of the page.
    1. If you make a mistake, hover over the x,y,z coordinates and select the trash bin. This will delete that point.
    1. To add a description to the point, go to the lower right side box and type in a description where it says 'Description'.
    You don't need to hit enter, simply move your mouse off that text input box. The description will appear below the x,y,z
       coordinates in the upper left side box.
    1. If you make lots of annotations, you might not see the new ones as they are added to the bottom. Simply expand
    the width of the box by hovering over the edge of the box (adjacent to the horizontal quadrant)
    1. If you want to change the size of the points, go to the 'Rendering' tab located near the top right. Move
    the slider bar to adjust the size. This will apply to all points in that layer. By making the points
       different colors and adjusting the size, you can make the different sets of points different and easier to distinguish.
    1. To create the COM layer, follow the above instructions. Make sure the name you type in the description box exactly
    matches a structure. 7n is different from 7N !
    1. For the COM layer, make sure you change the name of the layer to COM upper case.   
1. When you click within the layer box, that sets the layer as the active layer. Once a layer is active, you can control the settings with the 'Source', 'Render', 'Seg.', and 'Annotations' tabs on the right side of the program.

#### Top right
1. Name of URL - enter a name to describe what you are viewing. This name gets saved as the title in the database. 
   You must be logged into https://activebrainatlas.ucsd.edu/admin to save or update.
1. Saving - This will update the database with the name and data you are currently viewing. Click this whenever
   you want to save your view.
1. New - This will create a new insert into the database with the name and data you are currently viewing. You should
   only do this once to create a new view
1. JSON - This is marked by the '{}'. This displays a popup of all JSON data. This probably isn't very helpful to an end user.
1. Help - This is marked by a question mark. This popup will show all the keyboard shortcuts.
1. Wiki - This link points to this page.

##### Right side matrices
1. Source
    1. Top source matrix
    1. Bottom matrix
    1. Fetch rotation matrix   
1. Rendering
    1. opacity
    1. min
    1. max
    1. invert
    1. brightness
    1. contrast
    1. gamma
    1. linlog
1. Annotation
    1. color
    1. point
    1. bounding box
    1. line
    1. ellipsoid
    1. Export to CSV
    1. Import to CSV
    1. List of annotations 
       1. sorting
       1. x,y,z coordinates
       1. description
    1. Selection box in lower right
       1. previous selection
       1. pin selection
       1. close selection box   
       1. adding a description to an annotation
       
##### Command mouse and keyboard commands
1. Zoom in and out
1. Min max icons for each quadrant
1. Drag image through coordinates


