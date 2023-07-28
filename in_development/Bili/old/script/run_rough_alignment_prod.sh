#!/usr/bin/env bash
for brain in DK39 DK41 DK43 DK54 DK55
do
    ./run_rough_alignment.py $brain --dx 8 --dy 8 --dz 2 --niter 64
    ./run_rough_alignment.py $brain --dx 4 --dy 4 --dz 1 --niter 8 --cont
    ./run_rough_alignment.py $brain --dx 2 --dy 2 --dz 1 --niter 2 --cont --plot
done
