## Quality control process for each brain
1. For each channel, create a normalized and rotated image with this code:
    1. `mkdir -p ~/DK55/normalized`
    1. `cd /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK55/preps/CHX/thumbnail`
    1. Bili, you take C2T, channel 2 the red one, Junjie, you take C3T, channel 3 the green one    
    1. `for i in *.tif;do convert $i -normalize -rotate 270 -compress lzw ~/DK55/normalized/$i;done`
    1. download that ~/DK55/normalized dir to your local machine for easy viewing
    1. Go to https://activebrainatlas.ucsd.edu/activebrainatlas/admin/neuroglancer/urlmodel/
    1. Click on the DK55 link: 	"DK55 Low Res 3 Channels"
    1. Each z section corresponds to the image you downloaded above. e.g., section 030 is tif file: 030.tif
    1. Look for tissue missing near the brain stem and near the front (rostral) part of the brain.
    1. If you find any problems, report them in: https://activebrainatlas.ucsd.edu/activebrainatlas/admin/workflow/journal/
