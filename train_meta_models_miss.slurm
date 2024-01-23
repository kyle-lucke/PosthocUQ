#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o train_meta_models_miss_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch

echo "CNN/MNIST"
python train_meta_model_combine.py --base_model='SmallConvNetMNIST_BaseModel' --base_epoch=100 --meta_model='SmallConvNetMNIST_MetaModel_combine' --name='MNIST_miss' --dataset='MNIST' --lr=1e-1 --seed=0 --decay=5e-4 --epoch=50 --lambda_KL=1e-1 > mnist_meta_model.out
echo

# echo "CNN/SVHN"
# python train_meta_model_combine.py --base_model='SmallConvNetSVHN_BaseModel' --base_epoch=100 --meta_model='SmallConvNetSVHN_MetaModel_combine' --name='SVHN_miss' --dataset='SVHN' --lr=1e-1 --seed=0 --decay=5e-4 --epoch=50 --lambda_KL=1e-1 > svhn_meta_model.out
# echo

echo "VGG16/CIFAR10"
python train_meta_model_combine.py --base_model='VGG16_BaseModel' --base_epoch=250 --meta_model='VGG16_MetaModel_combine' --name='CIFAR10_miss' --dataset='CIFAR10' --lr=1e-4 --seed=0 --decay=5e-4 --epoch=50 --lambda_KL=1e-3 > cifar10_meta_model.out
echo

echo "VGG16/CIFAR100"
python train_meta_model_combine.py --base_model='VGG16_BaseModel' --base_epoch=170 --meta_model='VGG16_MetaModel_combine' --name='CIFAR100_miss' --dataset='CIFAR100' --lr=1e-1 --seed=0 --decay=1e-4 --epoch=50 --lambda_KL=1e-3 > cifar100_meta_model.out
echo 
