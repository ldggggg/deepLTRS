#!/bin/bash

module load conda/5.0.1-python3.6
source activate virt_pytorch
module load cuda/9.2
module load cudnn/7.1-cuda-9.2
module load gcc/7.3.0
module load mpi/openmpi-2.0.0-gcc
module load pytorch/1.4.0

#run train code
if [ $1 == "train" ]
then
    python TN_MB_Bias.py
#run inference code
elif [ $1 == "fix3" ]
then
    python TN_MB_Bias_fix3.py
	
elif [ $1 == "fix" ]
then
	python TN_MB_Bias_fix.py $1
    
else
    python train_texts.py
fi

