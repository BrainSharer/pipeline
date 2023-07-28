#!/usr/bin/env bash
./run_rough_alignment.py DK39 --dx 8 --dy 8 --dz 2 --niter 2
./run_rough_alignment.py DK39 --dx 4 --dy 4 --dz 1 --niter 2 --cont --plot
