## High level description of the preprocessing pipeline
1. To process a large stack of mouse brain images into a format that neuroscientists
can use requires a great deal of computational overhead. For the processing, 
the programming language pyython is used which has a large
user community and a rich set of libraries.
1. Our process makes extensive use of the following python libraries:
    1. [numpy](https://numpy.org/) Numpy takes the image data as arrays and is very
efficient at peforming many tasks on arrays of data.
    1. [opencv](https://opencv.org/) This library has many methods for image manipulation
    1. [pillow](https://python-pillow.org/) Another library for image manipulation
    1. [scikit image](https://scikit-image.org/) Another library for image manipulation
    1. [simpleITK](https://simpleitk.org/TUTORIAL/) A library for image registration,
analysis, segmentation and more.
    1. [Igneous labs cloud volume](https://github.com/seung-lab/igneous) This is
    the library that process the aligned images into Neuroglancer data.
    1. To get a full listing of all python libraries, see the 
[requirements.txt file](https://github.com/ActiveBrainAtlas2/preprocessing-pipeline/blob/master/requirements.txt) 
1. The entire process is run from one script `src/create_pipeline.py`
1. For instructions on running the pipeline see this 
[HOWTO](https://github.com/ActiveBrainAtlas2/preprocessing-pipeline/blob/master/docs/user/RUNNING.md)
### Raw images processing
1. The raw images that come of the scanning microscope are in a proprietary format called CZI. These files
are compressed and contain the images (around 20 per czi file) and also a great
deal of metadata that describes the scanner and also the images. This metadata
gets parsed with the [Bioformats tools](https://www.openmicroscopy.org/bio-formats/) 
from the CZI files and inserted into the MySQL database with and ORM library 
(Object relational mapping) called [sqlalchemy](https://www.sqlalchemy.org/).
1. After the TIF data has been extracted and the metadata inserted into the 
database, the user can verify the quality of the images and then proceed with
the pipeline. 
1. The next steps invole creating histograms for each file and downsampled
versions of the iamges that can be viewed on the web in a browser.

### Masking and cleaning
1. Masking is used to remove the shmuts that is in the clear areas outside the sections.
It is important to have clean images as they look better and it also makes
the alignment process more reliable. 

### Section to section alignment
1. After the images have been cleaned they are ready for alignment.
1. We use a tool called [Elastix](https://elastix.lumc.nl/). This tool
performs correlation between each adjoing image and returns a rotation, x-shift,
and y-shift for each consecutive set of images. This image is also stored
in the database in the elastix_transformation table.
1. The alignment is performed on the downsampled images as the full resolution
images would take too long to process. The full resolution images have the same
rotation but the x and y translations are multiplied by a scaling factor.

### Neuroglancer 
1. The aligned images are now ready to be processed into Neuroglancer's default image type: 
[precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)
1. This part of the pipeline makes extensive use of the 
[Igneous labs cloud volume](https://github.com/seung-lab/igneous) library.
1. The aligned stack of full resolution images is about 5-600GB in size. These 
files need to be broken down into *chunks* that Neuroglancer can use. The 
*cloud volume* library takes the aligned images and breaks them into the chunks.
These chunks of data get broken down into 9 different directories in the data 
directory. Each of these directories contains files that describe a different
resolution in Neuroglancer. The following directory structure describes the available 
resolutions with the first directory being the original resolution of the 
images in nanometers.

| Directory | Size | Number of files |
| ------- | ---- | ---- |
| 325_325_20000/ |  381GB |1,261,029 |
| 650_650_20000/ |  97GB | 316,820 | 
| 1300_1300_20000/ |  25GB | 79,716 | 
| 2600_2600_20000/ |  6.2GB | 19,929 | 
| 5200_5200_20000/ |  1.6GB | 5,180 | 
| 10400_10400_20000/ |  405MB | 1,330 | 
| 20800_20800_20000/ |  111MB | 1,330 | 
| 41600_41600_20000/ |  29MB | 350 |
| 83200_83200_40000/ |  4.2MB | 60 |

1. There are two steps to creating the precomputed format:
    1. Create the intial chunk size of (64,64,1). Neuroglancer serves data from 
    the webserver in chunks. The initial chunk only has a z length of 1. 
    This is necessary for the initial creation. However, this chunk size 
    results in too many files and needs to be *transfered* by the next step 
    in the process which creates a better chunk size and results in the *pyramid* 
    scheme that is best for viewing in a web browser. This data is stored 
    in */net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX/neuroglancer_data/CX_rechunkme*
    1. The 2nd phase in the precomputed process creates a set of optimum chunks from the 
    directory created in the previous step and places the new pyramid files in 
    */net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DKXX/neuroglancer_data/CX*
    This data is now ready to be served by the Apache web server. Note that 
    all the chunks (and there can be millions of files) are compressed with 
    *gzip* and so the Apache web server must be configured to serve compressed 
    files. This is done in one of the configuration files under the Apache 
    configuration directory on the web server.

### Cell Extraction
### Diffusion Map
### Training Detectors
### Brain to atlas alignment
#### Rough Alignment
#### Structure detection
#### fine alignment.
