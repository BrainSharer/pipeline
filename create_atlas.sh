#!/bin/bash

BASE="/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
python src/registration/scripts/create_registration.py --moving AtlasV8 --um 50 --task create_brain_coms
cp -vf $BASE/MD585/MD585_10um_sagittal.tif $BASE/AtlasV8/AtlasV8_10um_sagittal.tif
python src/registration/scripts/create_registration.py --moving MD585 --fixed AtlasV8 --um 50 --task register_volume
cp -vf $BASE/MD589/MD589_10um_sagittal.tif $BASE/AtlasV8/AtlasV8_10um_sagittal.tif
python src/registration/scripts/create_registration.py --moving MD589 --fixed AtlasV8 --um 50 --task register_volume
cp -vf $BASE/MD594/MD594_10um_sagittal.tif $BASE/AtlasV8/AtlasV8_10um_sagittal.tif
python src/registration/scripts/create_registration.py --moving MD594 --fixed AtlasV8 --um 50 --task register_volume
python src/registration/scripts/create_registration.py --moving AtlasV8 --um 50 --task create_average_volume
						
