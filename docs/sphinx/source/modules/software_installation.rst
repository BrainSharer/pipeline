Software installation
---------------------

Creating a sandbox on your computer or on a new workstation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. We are currently using Ubuntu 22.04 as of Oct 2023. Either install this on your 
   local machine or install it as a VM with Virtualbox or VMware. 

2. Cloning the repository and creating a virtual environment with our
   standard location:

- ``sudo apt install build-essential libmysqlclient-dev python3-dev``     
- ``git clone git@github.com:BrainSharer/pipeline.git``  
- ``sudo python3 -m venv /usr/local/share/pipeline``
- ``sudo chown -R $(id -u):$(id -g) /usr/local/share/pipeline``
- ``cd preprocessing-pipeline``
- ``source /usr/local/share/pipeline/bin/activate``
- ``pip install -U pip``
- ``pip install build``
- ``pip install -r requirements.txt``

3. Create this directory to start with:

- ``sudo mkdir -p /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK52/preps/CH1``

4. Make yourself user: ``sudo chown -R $(id -u):$(id -g) /net``
5. Get some thumbnails to start with:

- ``rsync -auv ratto.dk.ucsd.edu:/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK52/thumbnail_original/  /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK52/thumbnail_original/``
- ``rsync -auv ratto.dk.ucsd.edu:/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK52/preps/CH1/thumbnail/  /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK52/preps/CH1/thumbnail/``

6. You can now experiment with some of the thumbnails for DK52.
7. If you are using the VPN or are at UCSD, you won't need to install MariaDB, otherwise you will
   need to install the database portal yourself.
8. Clone the Django database portal repository, use the same virtualenv as above. You might
   need to install some more packages.

- ``git clone git@github.com:BrainSharer/BrainsharerMonoRepo.git``

Mysql for the database portal on Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  For complete instructions, look at this page:
   https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-22-04
   
Step-by-step guide:
~~~~~~~~~~~~~~~~~~~

1. Install and run mysql on your local machine. Again, you don’t need to
   do this if you are on the VPN or are at UCSD.

- ``sudo apt update && sudo apt install mariadb-server``
- ``sudo mysql_secure_installation``
- ``sudo mysql -u root -p``

2. Create a new user and a new database with the following SQL commands: 

- ``CREATE USER ‘dklab’@‘localhost’ IDENTIFIED BY ‘your_password_here’;`` 
- ``GRANT ALL ON active_atlas_development.* TO ‘dklab’@‘localhost’;``
- ``CREATE DATABASE active_atlas_development;``

3. Setup the database user by creating a file: ``~/.my.cnf`` in your
   home directory on your local machine:

- ``[client]`` 
- ``user = dklab``
- ``password = your_password_here``
- ``port = 3306``
- ``host = localhost``

4. Fetch the database with the last backup from ratto (to current
   directory), and import it to the database using the following BASH commands:

- ``last_backup=`ssh brainsharer.org ls -1tr /var/backups/brainsharer/brainsharer*.gz | tail -1``
- ``rsync -auv brainsharer.org:/var/backups/brainsharer/$last_backup ./``
- ``mysql brainsharer < $last_backup``


6. Test by going into the database and running some commands:

- ``mysql brainsharer``
- ``show tables;``


Here is a list of the software we use on a daily basis:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Visual Code - IDE for python and typescript.
- Dbeaver - database GUI tool
- imagemagick - used for converting images.
- matlab - we are not using this much. UCSD license is also available
- jupyter notebooks
- Fiji, port of ImageJ
- 3D Slicer
- Gimp - image editing software
- Geeqie - image viewer

Directory structure of the pipeline
===================================

1. The base directory is located on birdstore at:
   ``/net/birdstore/Active_Atlas_Data/data_root/pipeline_data``
2. All brains are located in the base directory.
3. The directory structure of a 3 channel brain is shown below. Data is served via Apache with the data in the
   neuroglancer_data directory. Each neuroglancer_data directory will usually contain a directory for each 3 channels.
   For example, under C1 there are several subdirectories representing a zoom level of the pyramid. The 325_325_20000 directory
   represents the an x,y,z resolution in nanometers. The x,y resolution is at 325nm and the z is at 20,000nm (20um). The structure
   for DK52 is:

::

   DK101
   ├── czi
   ├── preps
   │   ├── CH1
   │   │   ├── normalized
   │   │   ├── thumbnail
   │   │   ├── thumbnail_aligned
   │   │   ├── thumbnail_aligned_iteration_0
   │   │   ├── thumbnail_aligned_iteration_1
   │   │   ├── thumbnail_aligned_iteration_2
   │   │   └── thumbnail_cleaned
   │   ├── CH2
   │   │   ├── normalized
   │   │   ├── thumbnail
   │   │   ├── thumbnail_aligned
   │   │   ├── thumbnail_aligned_iteration_0
   │   │   ├── thumbnail_aligned_iteration_1
   │   │   ├── thumbnail_aligned_iteration_2
   │   │   └── thumbnail_cleaned
   │   ├── CH3
   │   │   ├── normalized
   │   │   ├── thumbnail
   │   │   ├── thumbnail_aligned
   │   │   ├── thumbnail_aligned_iteration_0
   │   │   ├── thumbnail_aligned_iteration_1
   │   │   ├── thumbnail_aligned_iteration_2
   │   │   └── thumbnail_cleaned
   │   ├── elastix
   │   ├── masks
   │   │   └── CH1
   │   ├── thumbnail_original
   │   └── tif
   └── www
     ├── neuroglancer_data
      │   ├── C1T
      │   ├── C2T
      │   └── C3T
      ├── scene
      └── section

   

Database backups
================

1. The development and production databases are backed up multiple times
   each day on basalis
2. If you need a backup, look on brainsharer.org at:
   ``/var/backups/brainsharer``
3. The development database is named ``active_atlas_development``
4. The production database is named ``brainsharer`` ###
   Setting up SSH connections to the servers
5. Refer `Checking for existing SSH
   keys <https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/checking-for-existing-ssh-keys>`__
   and `Generating a new SSH key and adding it to the
   ssh-agent <https://docs.github.com/en/enterprise-server@2.19/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`__
   for setting up the SSH on your local machine.
6. Substitute hostX and domainname names below with real names
7. Appending the following test in the SSH config file ``~/.ssh/config``
   to allow SSH server name aliasing

::

   Host host1
     HostName host1.domainname
     User <AD_username_here>

   Host host2
     HostName host2.domainname
     User <AD_username_here>

   Host host3
     HostName host3.domainname
     User <AD_username_here>

Then copy the SSH identity to the remote server, enter your AD password
when prompted.

.. code:: bash

   for server in host1, host2 host3; do
       ssh-copy-id -i $server
   done

Now you should be able to SSH into the servers without password.

Set up PYTHONPATH environmental variable
========================================

| the pythonpath environmental variable allows you to add folder to the
  search path of python automatically.
| This is useful for adding project folder to python path so that they
  work like normal packages in terms of imports. For the preprocessing
  project, the code lives in the src directory so you’ll want to add
  that path to your PYTHONPATH in your IDE

1. Install list of packages in requirements.txt
2. Install elastix, though we are using the SimpleITK version that
   includes elastix. 

Configuring imagemagick
=======================

Because imagemagick is not configured by default to work with large
images, we need to modify the policy file for imagemagick using the
following steps: after install imagemagick, use:

- ``identify -list policy | head`` 

to find out the path of the policy files do:

- ``sudo vim <path to policy.xml}/policy.xml``

 and change the following settings: 
 
 - 10 GB memory/disk limit
 - 500KP image size limits
   
These settings seem to be sufficient for microscopy images, but you can adjust them
depending on your image size and computational resources.

