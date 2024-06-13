Brain data ingestion processing steps
-------------------------------------

Prerequisites
~~~~~~~~~~~~~

*   **Note:** These instructions assume you previously completed imaging scans and have created a folder for your .czi files. This files should be stored in birdstore file server in directory:

*   Windows Example: Z:\Active_Atlas_Data\data_root\pipeline_data\{ANIMAL_ID}\czi\

*   Linux Example: /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{ANIMAL_ID}/czi/ 
    where {ANIMAL_ID} is given name of animal subject. Contact immediate supervisor for available names/sequences.

*   **N.B.** You will need to create czi folder under animal name and copy all .czi files into this directory 

---- 

1. Login


*   Navigate to BrainSharer Admin portal (brainsharer.org)

    a. Login in upper-right-hand corner

    b. After authentication, click username and choose ‘Admin’ from drop down menu 

.. image:: /_static/login.png


1.  Enter experimental subject information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create new **ANIMAL_ID** in the database using **Add** link under **BRAIN -> Animals**

.. image:: /_static/brain.categories.png

**ANIMAL_ID** is same name used in **PREREQUISITE** above (e.g. name of folder on birdstore)
Only **BOLD** fields are required **(Prep id)**; other fields are optional
    
.. |save| image:: /_static/save.png

Click |save| at the bottom of the form to proceed.

2.  Enter scanning information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add scanning info for **ANIMAL_ID** using ‘Add’ link under **BRAIN > Scan runs**

.. image:: /_static/scan_run.png

Notes:
~~~~~~

*   XY resolution is required (likely .325 µm on Axios)

*   Z resolution is required (likely 20 µm on Axios)

*   Number of slides (directory listing count for number of slides)

*   Width (pixels) (may leave at 0 – autodetect later)

*   Height (pixels) (may leave at 0 – autodetect later) 

Choose **ANIMAL_ID** in dropdown then enter details of scan. 

Click |save| at the bottom of the form to proceed.

3.  Enter histology information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **ANIMAL_ID** in dropdown then enter histology. 

.. image:: /_static/histology.png

Enter as much information as possible. The **bold** fields are required.

---- 

Congratulations! - The brain is now ready for processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Back-end steps for processing brains - Overview
===============================================

.. image:: /_static/pipeline.flow.png

Figure 1. Summary description of data ingestion pipeline. This describes the automated and manual steps and total automated time for 
representative brain with 3 channels (34 hours). 

Note: above steps are controlled through terminal python application, accessible on Linux. All files should be installed 
under /data/pipeline folder but may need updating if you encounter any errors. If script is not in this 
location, or you encounter errors, contact Duane or Ed for assistance. 

Login to Linux:
==================

Kleinfeld lab has 3 primary workstations used for brain processing with varying capacities: muralis (most powerful in cores/RAM), ratto, basalis. 
Brains will process faster on muralis, however other lab members may already be using the workstation.

Note: Use ActiveDirectory login to access all workstations.

Navigate to /data/pipeline/ (contains all scripts)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`cd /data/pipeline`

Activate python virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`source /usr/local/share/pipeline/bin/activate`

Run each requested step in sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`python src/pipeline/scripts/create_pipeline.py --task {TASK} --animal {ANIMAL_ID}`

*   Use ANIMAL_ID where czi images are located.

*   Default channel is 1

*   Available tasks are:

extract | mask | clean | align | histogram | neuroglancer | cell_labels | status | extra_channel

(but may need multiple runs if multiple channels, downsampled version is required or errors occur)

Note: When run with –-debug True , a log file will be created in root folder of processing folder on birdstore (pipeline-process.log). 
Check this file for status and/or errors. 
