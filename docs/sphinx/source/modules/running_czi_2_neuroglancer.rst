Running the entire process - HOWTO
----------------------------------

Setup
~~~~~

*   Use the installed virtual environment on ratto, basalis and
    muralis by running: ``source /usr/local/share/pipeline/bin/activate``

*   To run any of these commands at high resolution, prefix each command
    with ``nohup``\ and end the command with ``&``. That will make it run
    in the background and you can log out.

Running the initial task - extract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Create a folder with brain id under
    /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX create
    subfolder DKXX/czi and DKXX/tif. Then copy the czi files to
    DKXX/czi.
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
    #. After this is one you’ll want to do the Django database portal QC
       on the slides

Database portal QC after extract task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   Have someone confirm the status of each slide in:
    https://www.brainsharer.org/brainsharer/admin

    #.  After logging in, go to the Slides link in the Brain category.
    #.  Enter the animal name in the search box.
    #.  Under the PREP ID column, click the link of the slide you want
        to edit.
    #.  If the entire slide is bad or out of focus or is the very last scene, 
        mark it accordingly under it's Slide Status dropdown menu.
    #.  **When performing any operations on a scene, do them one at a time**
        and make use of the 'Save and continue' button. If a scene or scenes are bad, first
        mark it as bad, hit the 'Save and continue'. Do this individually for each scene.
        **Don't** try to mark multiple slides as Bad, and then hit 'Save and continue'.
    #.  If a scene needs to be marked as Bad, Out of Focus or the End
        slide, select the appropriate Scene QC. If you mark it as Bad or
        Out of Focus, the nearest neighbor will NOT be inserted. 
    #.  Do all scene replication after you have marked scenes bad. Make
        use of the ‘Save and continue’ button instead of the ‘Submit’
        button.
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

Running mask task
~~~~~~~~~~~~~~~~~

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


Running clean task
~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task clean``

    #. This will finalize the masks
    #. Creates the cleaned files from the masks

Running histogram task
~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task histogram``

    * This will create all the histograms

Running alignment task
~~~~~~~~~~~~~~~~~~~~~~

*   Run: ``python src/pipeline/scripts/create_pipeline.py --animal DKXX --task align``

    #. This will run elastix and create the rigid transformation for
       consecutive pair of files.
    #. Data is stored in the elastix_transformation table
    #. The alignment process is then run from the elastix data
    #. View the aligned images to make sure they all look good. ImageJ
       or geeqie is good for viewing lots of files.
    #. Some of the images might need manual alignment. Use the
       Manual_Alignment.ipynb jupyter notebook to do the manual
       alignment. Note, this will only work with sections before the
       midpoint. You’ll need to update the database with new parameters.
       The SQL is produced in the last cell of the notebook.

Running neuroglancer task
~~~~~~~~~~~~~~~~~~~~~~~~~

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
