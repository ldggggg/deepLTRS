#!/bin/bash

oarsub -p "gpucapability >= '5.0' and gpu='YES'" -l /gpunum=1,walltime=20:00:00 -S "./run_network.sh "$1


