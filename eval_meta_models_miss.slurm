#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o eval_meta_models_miss_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-gpu-dirichlet

# Evaluate LeNet meta-model for misclassification detection
# echo 'LeNet/MNIST'
# python eval_meta_model.py --base_model='LeNet_BaseModel' --meta_model='LeNet_MetaModel_combine' --name='MNIST_miss' --dataset='MNIST' --base_epoch=20 --seed 2
# echo 

echo "SVHN"
python eval_meta_model.py --base_model='SmallConvNetSVHN_BaseModel' --meta_model='SmallConvNetSVHN_MetaModel_combine' --name='SVHN_miss' --dataset='SVHN' --seed=0 > test_svhn_meta_model.out
echo

echo "Resnet32/CIFAR10"
python eval_meta_model.py --base_model='ResNetWrapper' --meta_model='Resnet_MetaModel_combine' --name='CIFAR10_miss' --dataset='CIFAR10' --seed=0 > test_resnet32_cifar10_meta_model.out
echo

# echo "WideResNet/CIFAR100"
# python eval_meta_model.py --base_model='WideResNet_BaseModel' --meta_model='WideResNet_MetaModel_combine' --name='CIFAR100_miss' --dataset='CIFAR100' --seed 1
# echo
