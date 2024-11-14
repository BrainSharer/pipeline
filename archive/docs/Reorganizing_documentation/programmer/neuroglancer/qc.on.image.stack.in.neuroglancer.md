## Neuroglancer QC

Stacks of downsampled images are set in Neuroglancer and need to be checked for quality before creating the full resolution stack. Errors occur during masking which sometimes removes important tissue. This often happens near the brain stem in the lower part of the image (the ventral and caudal area of the mouse brain). Each channel of the stack needs to be viewed and if there might be a problem, the image in Neuroglancer needs to be compared to the raw image from the pipeline process. The best way to accomplish this task is as follows:

1. Open up the 3 downsampled stacks in Neuroglancer.
1. Start with channel 1, and hide the remaining 2 channels.
1. Maximize the upper left quadrant (the sagittal view) in Neuroglancer.
1. Zoom into a level where the entire image takes up the screen.
1. Start at section 0 and go through each section. Any time a potential problem arises, compare it to the original image from the pipeline stack. 
1. The raw images (they have been histogram equalized and rotated, not masked) are on birdstore at: /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK60/preps/CH1/normalized substitute DK60 and CH1 for the animal and channel you want to view. You'll probably need the images on your local computer for easy viewing. Do something like:
`rsync -auv -e ssh myusername@hostname:/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK60/preps/CH1/normalized/ ./` That will copy all the files from the server to the current directory.
1. When you find a problem, it needs to be reported. You can report the problem in the Journal area of the database portal. This is at: https://activebrainatlas.ucsd.edu/activebrainatlas/admin/workflow/journal/
In the journal area when you add a problem report, you can also upload a screenshot. If you upload an image, it will only display web images, such as JPG or PNG. It won't display tif files. 
To actually fix a problem, the mask needs to be modified. This often means dilating it just a bit. This needs to be done manually with GIMP. The masks are on birdstore at: /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK60/preps/thumbnail_masked 
1. Repeat these steps on the remaining 2 channels. Note that if you find a masking problem on one channel, it very often will occur on the other 2 channels. The same mask is used on each channel.
