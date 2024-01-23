#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o train_base_models_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch

echo "MNIST/CNN"
python3 train_base_model.py --model='SmallConvNetMNIST_BaseModel' --name='MNIST' --dataset='MNIST' --seed=0 > mnist.out

echo "SVHN/CNN"
python3 train_base_model.py --model='SmallConvNetSVHN_BaseModel' --name='SVHN' --dataset='SVHN' --seed=0 > svhn.out

echo "VGG16/CIFAR10"
python train_base_model.py --model='VGG16_BaseModel' --name='CIFAR10' --dataset='CIFAR10' --seed=0 > cifar10.out
echo 

echo "VGG16/CIFAR100"
python train_base_model.py --model='VGG16_BaseModel' --name='CIFAR100' --dataset='CIFAR100' --seed=0 > cifar100.out
echo 

