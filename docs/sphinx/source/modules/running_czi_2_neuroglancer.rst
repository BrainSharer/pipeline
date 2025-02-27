Running the entire process - HOWTO
----------------------------------

Setup
~~~~~

*   Use the installed virtual environment on ratto, basalis and
    muralis by running: ``source /usr/local/share/pipeline/bin/activate``

*   To run any of these commands at high resolution, prefix each command
    with ``nohup``\ and end the command with ``> DKXX.log 2>&1 &``. That will make it run
    in the background, redirect loggging to DKXX.log  and you can log out.

Running the initial task - extract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Create a folder with brain id under
    /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX create
    subfolder DKXX/czi. Then copy the czi files to
    DKXX/czi. You could also link the files from the original origin folder. Copying
    czi files takes a long time.
*   Add entries in the animal, scan run and histology table in the
    database using the `admin portal <https://www.brainsharer.org/brainsharer/admin>`__.
*   All functionality for running the pipeline is found in
    src/pipeline/scripts/create_pipeline.py
*   You can run the script with the ‘h’ option to get the arguments:
    ``python src/pipeline/scripts/create_pipeline.py -h``
*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task extract``

    #. This will scan each czi file.
    #. Extracts the tif meta information and inserts into the
       slide_czi_to_tif.
    #. Also creates the downsampled tif files
    #. By default it works on channel 1 and the downsampled stack.
    #. Create png files for channel 1.
    #. After this task is done, you’ll want to do the Django database portal QC
       on the slides

Database portal QC after extract task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Have someone confirm the status of each slide in:
    https://brainsharer.org/brainsharer/admin

    #.  After logging in, go to the Slides link in the Brain category.
    #.  Enter the animal name in the search box.
    #.  Under the PREP ID column, click the link of the slide you want
        to edit.
    #.  If the entire slide is bad or out of focus or is the very last scene, 
        mark it accordingly under it's Slide Status dropdown menu.
    #.  When performing any operations on a scene, **do the operations one at a time**
        and make use of the 'Save and continue' button. If a scene or scenes are bad, first
        mark it as bad, hit the 'Save and continue'. Do this individually for each scene.
        **Don't** try to mark multiple slides as Bad, and then hit 'Save and continue'.
    #.  If a scene needs is no good, mark it as inactive at the bottom area of the page.
    #.  If you want to replicate a scene, choose the Replicate field and
        add an amount to replicate.
    #.  The list of scenes are listed near the bottom of the page.
    #.  If you make a mistake, you will need to reset the slide. To do this,
        click the 'Reset to original state' button at the very bottom of the page.
        This will set the slide and scenes back to their original state.
    #.  When you are done, click one of the Save buttons at the bottom
        of the page.
    #.  After editing the slides, you can view the list of sections
        under the Sections link under Brain. You will need to enter the animal
        name in the search field.

Running the mask task
~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task mask``

    #. This will read the sections view in the database.
    #. Creates the downsampled files
    #. Create the normalized files
    #. Creates the initial colored masks
    #. You’ll need to now verify the masks and possibly edit some of
       them in GIMP. Use color white to add to the mask and color black
       to subtract
    #. Check the rotation necessary for the images by viewing the
       normalized files and then updating the scan run table in the
       Django database portal and update the rotations. It is usually 3.
    #. The tighter the mask, the better the alignment will be.


Running the clean task
~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task clean``

    #. This will finalize the masks
    #. Creates the cleaned files from the masks

Running the histogram task
~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task histogram``

    * This will create all the histograms. You can view the histograms
      in the Django database portal under the Sections link. You can also view 
      the the histograms under the animal list page.

Running the alignment task
~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task align``

    #. This will run elastix and create the rigid transformation for
       consecutive pair of files.
    #. Data is stored in the elastix_transformation table
    #. The alignment process is then run from the elastix data
    #. View the aligned images to make sure they all look good. ImageJ
       or geeqie is good for viewing lots of files.
    #. Some of the images might need manual alignment. You will need to run the realign task.

Running the realignment task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task realign``

    #. This is necessary when the regular alignment doesn't work well enough. This is often
       due to bad or missing tissue.
    #. This task will first create two sets of neuroglancer data, one from the cropped data which is unaligned, 
       and one from the aligned data.
    #. First get a list of sections that are not aligned well. This can be done by viewing the aligned images 
       in Neuroglancer. After getting the sections, open up the Unaligned data as a separate layer in Neuroglancer.
       The unaligned data is available in the Neuroglancer precomputed area with https://imageserv.dk.ucsd.edu/data/DKXX/neuroglancer_data/C1T_unaligned
    #. Create a new annotations layer.
    #. Find common points within the adjacent sections. They must be the in the same place in the brain. Pick at least
       two points that are not too close. 
    #. If you have more than 2 adjacent sections, the points must all point to the same place in the brain.
    #. Label the annotations as Fiducial in the label search input field. (tab on the right side panel)
    #. Each adjacent section must have the **same number of points** and **be in the same location**.
    #. After you have marked all the fiducials, save the annotations with the 'Save' icon on the top-right side of the window.
    #. Go back to the pipeline and run the realign task. It should print information regarding finding points and realigning.
    #. You must now delete all the images under:
            
        * DKXX/preps/C1/thumbnail_aligned 
        * DKXX/www/neurglancer_data/C1T
        * DKXX/www/neuroglancer_data/C1T_rechunkme
        * DKXX/www/neuroglancer_data/progress/C1T

    #. Rerun the align task and then the neuroglancer task.
    

Running the neuroglancer task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task neuroglancer``

    #. This will create all the neuroglancer files in C1T_rechunkme and
       then C1T. C1T_rechunkme is the preliminary data directory that is
       created by the create_neuroglancer_image method and is then later
       used by the create_downsampling method.
    #. View results in neuroglancer. Add the layer to the precompute
       with:
       https://activebrainatlas.ucsd.edu/data/DKXX/neuroglancer_data/C1T

Running on other channels and the full resolution images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   After you have completed the downampled version for channel 1, you
    can repeat this entire process by running
    ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --channel 2``
    This will run the entire process for channel 2. Some of the tasks
    will be automatically skipped.
*   Repeat the process again for channel 3. Once you are happy with all
    the results, run the process again but with
    ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --channel 1 downsample false``.
    Some of the steps will be skipped automatically.

Final steps
~~~~~~~~~~~

*   After you have completed the steps above, make sure to create a symbolic
    link on imageserv.dk.ucsd.edu in the /srv directory pointing to /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX/www DKXX