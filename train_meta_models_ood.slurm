#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o train_meta_models_ood_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu-dirichlet

# # Train LeNet meta-model for OOD detection using LeNet base model
# echo "LeNet/MNIST"
# python train_meta_model_combine.py --base_model='LeNet_BaseModel' --base_epoch=20 --meta_model='LeNet_MetaModel_combine' --name='MNIST_OOD' --dataset='MNIST' --lr=1e-1 --seed=2 --decay=5e-4 --epoch=50 --lambda_KL=1e-1 
# echo

echo "VGG16/CIFAR10"
python train_meta_model_combine.py --base_model='VGG16_BaseModel' --meta_model='VGG16_MetaModel_combine' --name='CIFAR10_OOD' --dataset='CIFAR10' --lr=1e-3 --seed=5 --decay=1e-4 --epoch=50 --lambda_KL=1e-3 
echo 

# echo "WideResNet/CIFAR100"
# python train_meta_model_combine.py --base_model='WideResNet_BaseModel' --meta_model='WideResNet_MetaModel_combine' --name='CIFAR100_OOD' --dataset='CIFAR100' --lr=1e-2 --seed=2 --decay=1e-4 --epoch=50 --lambda_KL=1e-3

# paper params
# python train_meta_model_combine.py --base_model='WideResNet_BaseModel' --meta_model='WideResNet_MetaModel_combine' --name='CIFAR100_OOD' --dataset='CIFAR100' --lr=1e-1 --seed=1 --decay=1e-4 --epoch=50 --lambda_KL=1e-3
# echo 
