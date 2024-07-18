from __future__ import print_function

import os
import csv
import json
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics

import dataloaders as loaders
import models
import preproc as pre
from metrics import *

from model_utils import get_preprocessor

from utils import ROC_OOD, ROC_Selective, convert_to_rgb, set_seed

parser = argparse.ArgumentParser(description='Meta model Evaluation')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")


parser.add_argument('--base_model', default="WideResNet_BaseModel",
                    choices=('SmallConvNetSVHN_BaseModel', 'ResNetWrapper'),
                    type=str, help='model type (default: LeNet)')

parser.add_argument('--meta_model', default="WideResNet_MetaModel_combine", type=str,
                    choices=('SmallConvNetSVHN_MetaModel_combine', 'Resnet_MetaModel_combine'),
                    help='model type (default: LeNet)')

parser.add_argument('--name', default='CIFAR100_OOD', choices=('CIFAR10_miss', 'SVHN_miss'),
                    type=str, help='name of run')

parser.add_argument('--dataset', default='CIFAR10', choices=('CIFAR10', 'SVHN'), type=str,
                    help='Dataset to use for run')

parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'SVHN':
    args.fea_dim = [8192, 4096, 2048]
# elif args.dataset == 'MNIST':
#     args.fea_dim = [5408, 9216]
# elif args.dataset == 'CIFAR10' and args.base_model == 'VGG16_BaseModel':
#     args.fea_dim = [16384, 8192, 4096, 2048, 512, 10]
elif args.dataset == 'CIFAR10' and args.base_model == 'ResNetWrapper':
    args.fea_dim = [8192, 8192, 4096, 4096, 64, 10]    
# elif args.dataset == 'CIFAR100':
#     args.fea_dim = [16384, 8192, 4096, 2048, 512, 100]

set_seed(args.seed, use_cuda)


'''
Processing data
'''
print('==> Preparing data..')
# # Noisy validation set for OOD
# if args.dataset == 'MNIST':
#     transform_noise = transforms.Compose([
#         transforms.Resize(32),
#         transforms.Lambda(convert_to_rgb),
#         transforms.ToTensor(),
#         pre.PermutationNoise(),
#         pre.GaussianFilter(),
#         pre.ContrastRescaling(),
#     ])
# else:
#     transform_noise = transforms.Compose([
#         transforms.Resize(32),
#         transforms.ToTensor(),
#         pre.PermutationNoise(),
#         pre.GaussianFilter(),
#         pre.ContrastRescaling(),
#     ])

testloader = None
if args.dataset == 'CIFAR10':
    
    print('CIFAR10')

    _, transform_test = get_preprocessor(args.dataset)

    dataset = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                            transform=transform_test)

    val_idxs = np.load('trained-base-models/cifar10-resnet32/val_idxs.npy')

    valset = data.Subset(dataset_val, val_idxs)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=8)

    
    testset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=False,
                               transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             num_workers=8)

elif args.dataset == 'SVHN':

    _, transform_test = get_preprocessor(args.dataset)

    dataset = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                            transform=transform_test)
    
    val_idxs = np.load('trained-base-models/svhn-cnn/val_idxs.npy')

    valset = data.Subset(dataset, val_idxs)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=8)
    
    testset = datasets.SVHN(root='~/data/SVHN', split='test', download=True,
                            transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             num_workers=8)

# FIXME: OOD data should be normalized same way as ID data
# transform_ood_color = transforms.Compose([
#     transforms.Resize(32),
#     transforms.Lambda(convert_to_rgb),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# transform_ood_normal = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
    
# '''
# Processing OOD data
# '''
# oodloaders = []
# if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
#     oodnames = ['SVHN', 'FashionMNIST', 'LSUN', 'TinyImage', 'Corrupted']
#     oodloaders.append(loaders.SVHN(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.FashionMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.LSUN_CR(train=False, batch_size=100, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
#     oodloaders.append(loaders.TinyImageNet(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.Corrupted(args.dataset, transform_noise, batch_size=100, shuffle=False, num_workers=8))
# elif args.dataset == 'MNIST':
#     oodnames = ['Omniglot', 'FashionMNIST', 'KMNIST', 'CIFAR10', 'Corrupted']
#     oodloaders.append(loaders.Omniglot(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.FashionMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.KMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.CIFAR10(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
#     oodloaders.append(loaders.Corrupted(args.dataset, transform_noise, batch_size=100, shuffle=False, num_workers=8))

print('current seed', args.seed)

set_seed(args.seed, use_cuda)
    
print('==> Loading base model and meta model from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

# use models trained under our framework.
if args.dataset == 'CIFAR10':
  checkpoint_base = os.path.join('trained-base-models', 'cifar10-resnet32', 'best.pth')

elif args.dataset == 'SVHN':
  checkpoint_base = os.path.join('trained-base-models', 'svhn-cnn', 'best.pth')


# Load base model and meta model
checkpoint_base = torch.load(checkpoint_base, map_location=device)

checkpoint_meta = torch.load(f'checkpoint/{args.name}_{args.meta_model}_{args.seed}.ckpt',
                             map_location=device)

# get base net
if args.dataset == 'CIFAR10':
    base_net = models.__dict__[args.base_model](10)
else:
    base_net = models.__dict__[args.base_model]()

base_net.load_state_dict(checkpoint_base)
base_net.eval()


# get meta net
if args.dataset == 'CIFAR10' and args.base_model == 'ResNetWrapper':
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0],
                                                fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2],
                                                fea_dim4=args.fea_dim[3],
                                                fea_dim5=args.fea_dim[4],
                                                n_classes=args.fea_dim[5])
elif args.dataset == 'SVHN':
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0],
                                                fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2])

meta_net.load_state_dict(checkpoint_meta['meta_net'])
meta_net.eval()


# transfer models to gpu if applicable    
if use_cuda:
    base_net = base_net.cuda()
    meta_net = meta_net.cuda()
    cudnn.benchmark = True

print("Base net is on device:", next(base_net.parameters()).device)
print("Meta model is on device:", next(meta_net.parameters()).device)

# Collect all uncertainty scores
def get_uncertainty_score(loader, label, get_preds=False):
    assert label in [0, 1]
    label = 1. * label

    base_net.eval()
    meta_net.eval()

    labels = []  # is this point ID (0) or OOD (1)?
    base_ents = []
    base_maxps = []
    base_energy = []

    diff_ents = []
    mis = []
    ents = []
    maxps = []
    precs = []

    base_preds = []
    preds = []

    with torch.no_grad():
        # In distribution data
        for batch_idx, (xs, ys) in enumerate(loader):
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            base_logits, fea_list = base_net(xs)
            logits = meta_net(*fea_list)

            if get_preds:
                # Get predictions (misclassification binary labels)
                _, base_predicted = torch.max(base_logits.data, 1)
                base_wrongs = base_predicted.ne(ys.data)
                _, meta_predicted = torch.max(logits.data, 1)

                # meta_wrongs = meta_predicted.ne(ys.data)
                meta_correct = meta_predicted.eq(ys.data)

                base_preds.append(base_wrongs.data.cpu())
                
                # preds.append(meta_wrongs.data.cpu())
                preds.append(meta_correct.data.cpu())

                
            # Uncertainty Criterion
            labels.append(label * torch.ones(ys.shape[0]))

            base_ents.append(compute_total_entropy(base_logits).data.cpu())
            base_maxps.append(compute_max_prob(base_logits).data.cpu())
            base_energy.append(torch.logsumexp(base_logits, -1).data.cpu())

            diff_ents.append(compute_differential_entropy(logits).data.cpu())
            mis.append(compute_mutual_information(logits).data.cpu())
            ents.append(compute_total_entropy(logits).data.cpu())
            maxps.append(compute_max_prob(logits).data.cpu())
            precs.append(compute_precision(logits).data.cpu())

        if get_preds:
            base_preds = torch.cat(base_preds, 0)
            preds = torch.cat(preds, 0)

        return torch.cat(diff_ents, 0), \
               torch.cat(mis, 0), \
               torch.cat(ents, 0), \
               torch.cat(maxps, 0), \
               torch.cat(precs, 0), \
               torch.cat(labels, 0), \
               torch.cat(base_ents, 0), torch.cat(base_maxps, 0), torch.cat(base_energy, 0), \
               base_preds, preds


def determine_threshold(max_threshold_step=.01):

    # NOTE: the actual threhsold step may be slightly lower than
    # max_threshold_step due to roundoff
    
    _, _, _, max_p, _, _, _, _, _, _, meta_preds = get_uncertainty_score(valloader,
                                                                         label=0, get_preds=True)

    misclf_labels = meta_preds.int().detach().cpu().numpy()
    max_p = max_p.detach().cpu().numpy()

    # determine how many elements we need for a pre-determined spacing
    # between thresholds. taken from:
    # https://stackoverflow.com/a/70230433
    num = round((max_p.max() - max_p.min()) / max_threshold_step) + 1 
    thresholds = np.linspace(max_p.min(), max_p.max(), num, endpoint=True)

    # compute performance over thresholds
    threshold_to_metric = {}
    for tau in thresholds:

        predicted_labels = threshold(max_p, tau)

        tn, fp, fn, tp = metrics.confusion_matrix(misclf_labels, predicted_labels).ravel()
        
        specificity_value = specificity(tn, fp)
        sensitivity_value = sensitivity(tp, fn)

        f_beta_spec_sens = f_score_sens_spec(sensitivity_value,
                                             specificity_value, beta=1.0)

        # print(f'tau: {tau}, spec: {specificity_value}, sens: {sensitivity_value}, f_beta: {f_beta_spec_sens}')
        
        threshold_to_metric[tau] = f_beta_spec_sens

        
    # print('threshold: f_beta_spec_sens')
    # for t, val in threshold_to_metric.items():
    #     print(f'{t}: {val}')
    # print(end='\n\n')

    # determine best threshold:
    best_item = max(threshold_to_metric.items(), key=lambda x: x[1])

    # print(best_item)

    return best_item[0]
    
threshold = determine_threshold()

print(f'selected threshold: {threshold}')
print()

diff_ents, mis, ents, maxps, precs, labels, base_ents, base_maxps, base_energy, base_preds, meta_preds = \
    get_uncertainty_score(testloader, label=0, get_preds=True)

# setup results directory
out_dir = os.path.join('results', f'eval_{args.name}_{args.meta_model}_{args.seed}')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# evaluation
if args.name in ['CIFAR10_miss', 'CIFAR100_miss', 'MNIST_miss', 'SVHN_miss']:
    
    # Evaluate misclassification performance (for test dataset)
    res = ROC_Selective(ents, maxps, meta_preds, threshold)
    
    for k, v in res.items():
        print(f'{k}: {v:.4f}')
    
    # save results:
    json.dump(res, open(os.path.join(out_dir, 'test_metrics.json'), 'w'))
        
# elif args.name in ['CIFAR10_OOD', 'CIFAR100_OOD', 'MNIST_OOD']:

#     print('FIXME: NEED TO CHANGE LABEL VALUES/METRIC VALUES FOR OOD)
#     exit()

#     for i in range(len(oodloaders)):
#         print(oodnames[i])
#         ood_diff_ents, ood_mis, ood_ents, ood_maxps, ood_precs, ood_labels, \
#         ood_base_ents, ood_base_maxps, ood_base_energy, *_ = \
#             get_uncertainty_score(oodloaders[i], label=1, get_preds=False)

#         all_diff_ents = torch.cat([diff_ents, ood_diff_ents])
#         all_mis = torch.cat([mis, ood_mis])
#         all_ents = torch.cat([ents, ood_ents])
#         all_maxps = torch.cat([maxps, ood_maxps])
#         all_precs = torch.cat([precs, ood_precs])
#         all_labels = torch.cat([labels, ood_labels])
#         all_base_ents = torch.cat([base_ents, ood_base_ents])
#         all_base_maxps = torch.cat([base_maxps, ood_base_maxps])

#         # Evaluate OOD detection performance
#         res = ROC_OOD(all_diff_ents, all_mis, all_ents, all_maxps, all_precs, all_labels,
#                       all_base_ents, all_base_maxps)
        
#         auroc_ent = res[0][0]
#         auroc_maxp = res[0][1]
#         auroc_mi = res[0][2]
#         auroc_dent = res[0][3]
#         auroc_prec = res[0][4]

#         aupr_ent = res[1][0]
#         aupr_maxp = res[1][1]
#         aupr_mi = res[1][2]
#         aupr_dent = res[1][3]
#         aupr_prec = res[1][4]

#         auroc_base_ent = res[2][0]
#         auroc_base_maxp = res[2][1]

#         aupr_base_ent = res[2][2]
#         aupr_base_maxp = res[2][3]

#         # save results:
#         log_name = os.path.join(out_dir, f'{oodnames[i]}.csv')

#         with open(log_name, 'w') as logfile:
#             logwriter = csv.writer(logfile, delimiter=',')
#             logwriter.writerow(['AUROC ENT', 'AUROC MAXP', 'AUROC MI', 'AUROC DENT',
#                                 'AUROC PREC', 'AUPR ENT', 'AUPR MAXP', 'AUPR MI', 'AUPR DENT',
#                                 'AUPR PREC', 'AUROC BASE ENT', 'AUROC BASE MAXP',
#                                 'AUPR BASE ENT', 'AUPR BASE MAXP'])
            
#             logwriter.writerow([auroc_ent, auroc_maxp, auroc_mi, auroc_dent, auroc_prec,
#                                 aupr_ent, aupr_maxp, aupr_mi, aupr_dent, aupr_prec,
#                                 auroc_base_ent, auroc_base_maxp, aupr_base_ent, aupr_base_maxp])
