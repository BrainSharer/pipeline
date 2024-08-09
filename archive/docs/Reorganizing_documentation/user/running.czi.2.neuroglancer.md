### Setup
1. To install the software prerequisites, [look here.](../programmer/preprocessing-pipeline/software.installation.md) 
Or use the installed virtual environment on ratto, basalis and muralis by running: 
```source /usr/local/share/pipeline/bin/activate```

1. To run any of these commands at high resolution, prefix each command with `nohup`and end the command with `&`. That will make it run in the background and you can log out.
1. All scripts that can be parallelized are done in the script with the get_cpus method.
   
### Process
1. Create a folder with brain id under /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX create subfolder DKXX/czi and DKXX/tif. 
Then copy the czi files to DKXX/czi.   
1. Add entries in the animal, scan run and histology table in the database using the admin portal.  
You can do this by using the corresponding app and clicking the button on the top right in the admin portal to create rows.
1. All functionality for running the pipeline is found in src/create_pipeline.py   
1. Run: `python src/create_pipeline.py --animal DKXX`
    1. This will scan each czi file.
    1. Extracts the tif meta information and inserts into the slide_czi_to_tif.
    1. Also creates the tif files
    1. By default it works on channel 1 and the downsampled stack.
    1. After this is one you'll want to do the Django database portal QC on the slides
1. Have someone confirm the status of each slide in: https://activebrainatlas.ucsd.edu/activebrainatlas/admin
    1. After logging in, go to the Slides link in the Brain category.
    1. Enter the animal name in the search box.
    1. Under the PREP ID column, click the link of the slide you want to edit.
    1. If the entire slide is bad, mark it as Bad under Slide Status.
    1. If a scene needs to be marked as Bad, Out of Focus or the End slide, select the appropriate Scene QC.
    If you mark it as Bad or Out of Focus, the nearest neighbor will NOT be inserted. If you need to replicate
    a scene, you use replicate scene functionality in the appropriate scene in the form.
    1. If you want to replicate a scene, choose the Replicate field and add an amount to replicate.
    1. The list of scenes are listed near the bottom of the page.
    1. When you are done, click one of the Save buttons at the bottom of the page.
    1. After editing the slides, you can view the list of sections under the Sections link under Brain.
    
1. Run: `python src/create_pipeline.py --animal DKXX step 1`
    1. This will read the sections view in the database. 
    1. Creates the full and downsampled files
    1. Create the normalized files
    1. Create png files for channel 1.
    1. Creates the final masks
    1. You'll need to now verify the masks and possibly edit some of them in GIMP.  Use color white to add to the mask and color black to subtract
    1. Check the rotation necessary for the images by viewing the normalized files and then 
    updating the scan run table in the Django database portal and update the rotations. It is usually 3.

1. Run: `python src/create_pipeline.py --animal DKXX --step 2`
    1. This will finalize the masks
    1. Creates the cleaned files from the masks
    1. Creates all the necessary histograms

1. Run: `python src/create_pipeline.py --animal DKXX --step 3`
    1. This will run elastix and create the rigid transformation for consecutive pair of files.
    1. Data is stored in the elastix_transformation table
    1. The alignment process is then run from the elastix data
    1. View the aligned images to make sure they all look good. ImageJ or geeqie 
    is good for viewing lots of files.
    
1. Run: `python src/create_pipeline.py --animal DKXX --step 4`
    1. This will create all the neuroglancer files in C1T_rechunkme and then C1T.
    C1T_rechunkme is the preliminary data directory that is created by the
    create_neuroglancer_image method and is then later used by the create_downsampling
    method.
    1. View results in neuroglancer. Add the layer to the precompute with:
        https://activebrainatlas.ucsd.edu/data/DKXX/neuroglancer_data/C1T
1. After you have completed the downampled version for channel 1, you can repeat
this entire process by running `python src/create_pipeline.py --animal DKXX --channel 2 --step 4`
This will run the entire process for channel 2. Most the steps will be automatically skipped.
1. Repeat the process again for channel 3. Once you are happy with all the results run
the process again but 
with `python src/create_pipeline.py --animal DKXX --channel 1 downsample false --step 4`
