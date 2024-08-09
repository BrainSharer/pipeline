### Usability checklist

Here is a list of things to check after deploying software to the server

## Test Neurogolancer-Django interaction
1. login to https://activebrainatlas.ucsd.edu/activebrainatlas/admin
2. Under 'NEUROGLANCER', Click 'Neuroglancer states' or click following link: https://activebrainatlas.ucsd.edu/activebrainatlas/admin/neuroglancer/urlmodel/ 
3. After new page loads, Search for **test_save_annotation**. 
4. Click 'test_save_annotation' link under 'NEUGOLANCER' heading [to open Neuroglancer with this view]
5. Click 'New' in Neuroglancer window [to create a new "view"]
6. create a new volume (Landmark optional)
  6a. switch to edit mode [for volume]
  6b. add points to volume
  6c. move points around [that belong to volume]
  6d. remove some points from volume
  6e. move entire contour around [shift left click]
  6f. duplicate entire contour [select contour + CTRL-C] *New pology should appear in annotations section with z value incremented by 1
  6g. delete entire volume
  6h. close current active volume session
7. Create new cell session (Description and category optional)
  7a. add point on image
  7b. swith to edit mode [CTRL X]
  7c. move point on image
  7d. remove point from image ((Ctrl + Alt + right click) or Trash can) 
  7e. close the session
8. Create new com session ('Description' optional)
  7a. add point on image
  7b. swith to edit mode [CTRL X]
  7c. move point on image
  7d. remove point from image ((Ctrl + Alt + right click) or Trash can) 
  7e. close the session
9. click 'save annotations' (under drop downs)
10. Create new annotation layer (tab) with [Ctrl + left click on plus sign]
11. Import the saved annotations from the drop down [the same volume, cell, com from steps 6-8]
12. Verify volume, cell, com are same as previous (locations)
  12a. Open one neuroglancer state with multisection annotation, click save and check for successful save message and database entry

### CLEANUP * check and delete stored annotations from admin portal
13. Return to Admin portal (leave Neuroglancer page open)
14. Click 'Annotation sessions' link under 'NEUGOLANCER' heading
15. Search for name of annimal and user name [steps 6-8].
16. Confirm the volume, cell and com sessions exist (3 entries) on page (Click 'Data' under 'SHOW POINTS' column for detail)
17. Select annotations checkboxes and select 'Delete selected Annotations sessions' from action dropdown; click 'Go'
    * You will see 'Successfully deleted {#} Annotations sessions' at top of screen (in green)  
18. Under 'NEUROGLANCER', Click 'Neuroglancer states'
19. Search for **test_save_annotation** (select recently created Neuroglancer states)
20. Select annotations checkboxes and select 'Delete selected Neuroglancer state' from action dropdown; click 'Go'
    * You will see 'Successfully deleted {#} Neuroglancer state' at top of screen (in green)  


## Test Slide QC
1. open the slide option in the brain app
2. search for "test"
3. click into a test entry
4. Mark 'Scene 1 QC' as 'bad tissue' (dropdown); click "save and continue editing"
  * When page refreshes, Scene 1 should not be visible
5. enable scene one again (select 'ok' from dropdown) click "save and continue"
  * When page refreshes, Scene 1 should be visible
6. Change 'Replicate S1' to '3'; click "save and continue editing"
  * When page refreshes, Check scene is duplicated 3 times
8. set scene one duplication to 0 again and click "save and continue"
  * When page refreshes, Check scene should appear only once

## Test Image Load:
1. Opening at 3 different DK brains and scroll through z stacks, check on channel color widget and intensity rendering.  Note down slow down in performances or crashes.
