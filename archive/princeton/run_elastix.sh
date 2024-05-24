
elastix -f $HOME/.brainglobe/sagittal_atlas_20um_iso.tif \
-m /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/CH1/image_stack.tif \
-out /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/CH1/registration \
 -p $HOME/programming/preprocessing-pipeline/pipeline/registration/princeton/parameterfolder/Order2_Par0000bspline.txt \
 -p $HOME/programming/preprocessing-pipeline/pipeline/registration/princeton/parameterfolder/Order1_Par0000affine.txt
