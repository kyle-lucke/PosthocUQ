from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn import metrics

import models
import preproc as pre
from losses import BeliefMatchingLoss
from metrics import compute_total_entropy, compute_max_prob, compute_differential_entropy, compute_mutual_information, \
    compute_precision
from utils import progress_bar, convert_to_rgb, set_seed
from model_utils import *

parser = argparse.ArgumentParser(description='Meta model training')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')

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
parser.add_argument('--epoch', default=20, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lambda_KL', default=1e-3, type=float, help='lambda for KL term in ELBO loss')
parser.add_argument('--verbose', action='store_true', help='Weather or not to use progress bar during model training (default: False).')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

args.augment = False

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_auroc = 0

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
# Noisy validation set for OOD
if args.dataset == 'MNIST':
    transforms.Resize(28),
    transform_noise = transforms.Compose([
        transforms.ToTensor(),
        pre.GaussianFilter(),
    ])
else:
    transform_noise = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        pre.PermutationNoise(),
        pre.GaussianFilter(),
        pre.ContrastRescaling(),
    ])

if args.dataset == 'CIFAR10':
    print('CIFAR10')

    _, transform_test = get_preprocessor(args.dataset)
    
    dataset = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=True,
                               transform=transform_test)
    
    dataset_val = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=False,
                                   transform=transform_test)

    dataset_noise = datasets.CIFAR10(root='~/data/CIFAR10', train=True, download=False,
                                     transform=transform_noise)

    # load indices used during our training
    train_meta_idxs = np.load('trained-base-models/cifar10-resnet32/trainset_meta_idxs.npy')
    val_idxs = np.load('trained-base-models/cifar10-resnet32/val_idxs.npy')

    trainset = data.Subset(dataset, train_meta_idxs)
    valset = data.Subset(dataset_val, val_idxs)
    valset_noise = data.Subset(dataset_noise, val_idxs)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=8)
    
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=8)


# elif args.dataset == 'MNIST':

#     transform_train, transform_test = get_preprocessor(args.dataset)
    
#     dataset = datasets.MNIST(root='~/data/MNIST', train=True, download=True, transform=transform_train)
#     dataset_val = datasets.MNIST(root='~/data/MNIST', train=True, download=True, transform=transform_test)
#     dataset_noise = datasets.MNIST(root='~/data/MNIST', train=True, download=False, transform=transform_noise)
    
#     train_list = np.load('dataset_idxs/mnist/train_idx.npy')
#     val_list = np.load('dataset_idxs/mnist/val_idx.npy')

#     trainset = data.Subset(dataset, train_list)
#     valset = data.Subset(dataset, val_list)
#     valset_noise = data.Subset(dataset_noise, val_list)

#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
#     valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
#     valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size, shuffle=False,
#                                                   num_workers=8)
#     testset = datasets.MNIST(root='~/data/MNIST', train=False, download=False, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


elif args.dataset == 'SVHN':

    _ , transform_test = get_preprocessor(args.dataset)
    
    dataset = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                            transform=transform_test)
    
    dataset_val = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                                transform=transform_test)
    
    dataset_noise = datasets.SVHN(root='~/data/SVHN', split='train', download=True,
                                  transform=transform_noise)

    # load indices used during our training
    train_meta_idxs = np.load('trained-base-models/svhn-cnn/trainset_meta_idxs.npy')
    val_idxs = np.load('trained-base-models/svhn-cnn/val_idxs.npy')

    trainset = data.Subset(dataset, train_meta_idxs)
    valset = data.Subset(dataset, val_idxs)
    valset_noise = data.Subset(dataset_noise, val_idxs)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8)
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=8)
    
    valloader_noise = torch.utils.data.DataLoader(valset_noise, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=8)


print(f'training on {len(trainset)} data samples.')
print(f'validating on {len(valset)} data samples.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Preparing model
'''
# if args.dataset == 'CIFAR100':
#     base_net = models.__dict__[args.base_model](100)
# elif args.dataset == 'CIFAR10':
#     base_net = models.__dict__[args.base_model](10)
# else:
#     base_net = models.__dict__[args.base_model]()

# if args.dataset in ['CIFAR10', 'CIFAR100']:
#     meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
#                                                 fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
#                                                 fea_dim5=args.fea_dim[4], n_classes=args.fea_dim[5])
# elif args.dataset == 'SVHN':
#     meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
#                                                 fea_dim3=args.fea_dim[2])

# else:
#     meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1])

# get base net
if args.dataset == 'CIFAR10':
    base_net = models.__dict__[args.base_model](10)
else:
    base_net = models.__dict__[args.base_model]()


# get meta net
if args.dataset == 'CIFAR10' and args.base_model == 'ResNetWrapper':
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
                                                fea_dim5=args.fea_dim[4], n_classes=args.fea_dim[5])
elif args.dataset == 'SVHN':
    meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                fea_dim3=args.fea_dim[2])

# transfer models to device
if use_cuda:
    base_net = base_net.cuda()
    meta_net = meta_net.cuda()

    print('Using CUDA..')
    print(torch.cuda.device_count())
    cudnn.benchmark = True

print("Base net is on device:", next(base_net.parameters()).device)
print("Meta model is on device:", next(meta_net.parameters()).device)

# FIXME: write resnet32 base-model / meta-model -- for probe placement
# place after maxpooling layers

# use models trained under our framework.
if args.dataset == 'CIFAR10':
  model_pth = os.path.join('trained-base-models', 'cifar10-resnet32', 'best.pth')

elif args.dataset == 'SVHN':
  model_pth = os.path.join('trained-base-models', 'svhn-cnn', 'best.pth')

# Load base net state dictionary
base_net.load_state_dict(torch.load(model_pth, map_location=device))
base_net.eval()
for k, v in base_net.named_parameters():
    v.requires_grad = False

meta_net.eval()

optimizer = optim.SGD(meta_net.parameters(), momentum=0.9, weight_decay=args.decay, lr=args.lr)

if not os.path.isdir('results'):
    os.mkdir('results')

logname = f'results/log_{meta_net.__class__.__name__}_{args.name}_{args.seed}.csv'

'''
Training Meta-model
'''
vi_loss = BeliefMatchingLoss(args.lambda_KL, 1)

def compute_logits_and_loss(xs, ys, compute_loss=False):
    loss = torch.Tensor([0])
    _, fea_list = base_net(xs)
    logits = meta_net(*fea_list)

    if compute_loss:
        loss = vi_loss(logits, ys)

    return logits, loss


def train(epoch):
    print('\nEpoch: %d' % epoch)
    base_net.eval()
    meta_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (xs, ys) in enumerate(trainloader):
        total += ys.size(0)

        if use_cuda:
            xs, ys = xs.cuda(), ys.cuda()

        logits, loss = compute_logits_and_loss(xs, ys, compute_loss=True)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        _, predicted = torch.max(logits.data, 1)
        correct += predicted.eq(ys).cpu().sum()
 
        if args.verbose:
            progress_bar(current=batch_idx, total=len(trainloader),
                         msg='Loss: %.3f |  Acc: %.3f%% (%d/%d)' %
                         (train_loss/(batch_idx + 1), 100*correct/total, correct, total))

    if not args.verbose:
      print('Train | Loss: %.3f |  Acc: %.3f%% (%d/%d)' %
            (train_loss/len(trainloader), 100*correct/total, correct, total))

    train_loss_final = train_loss/batch_idx
    acc = 100*correct/total

    return (train_loss_final, acc)


# '''
# Testing Meta-model
# '''

def validate(epoch):

    base_net.eval()
    meta_net.eval()
    valid_loss = 0
    correct = 0

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(valloader):
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            logits, loss = compute_logits_and_loss(xs, ys, compute_loss=True)
            valid_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += predicted.eq(ys.data).cpu().sum()

            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()

            if args.verbose:
                progress_bar(batch_idx, len(valloader),
                             'Loss: %.3f | Acc: %.3f%% (%d/%d) | DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
                             (valid_loss/(batch_idx + 1), 100*correct/total, correct, total,
                              diff_entropy/total, mutual_info / total,
                              total_entropy/total, max_prob/total, precision/total))

        if not args.verbose:
            print('Valid Metrics | Loss: %.3f | Acc: %.3f%% (%d/%d) | DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
                  (valid_loss/len(valloader), 100*correct/total, correct, total,
                   diff_entropy/total, mutual_info/total,total_entropy/total,
                   max_prob/total, precision/total))
            
        valid_loss_final = valid_loss/total
        acc = 100*correct/total

        return (valid_loss_final, acc)


'''
Validation for OOD task
'''
def UQ_validation():
    global best_auroc
    base_net.eval()
    meta_net.eval()
    flag = True
    total = 0
    with torch.no_grad():
        # In distribution data
        for batch_idx, (xs, ys) in enumerate(valloader):
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            logits, _ = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            mutual_info = compute_mutual_information(logits)
            _, meta_predicted = torch.max(logits.data, 1)
            meta_correct = meta_predicted.ne(ys.data)
            if flag:
                all_label = torch.zeros((ys.size()[0]))
                all_mutual_info = mutual_info.data.cpu()
                all_meta_predicted = meta_correct.data.cpu()
                flag = False
            else:
                all_label = torch.cat((all_label, torch.zeros((ys.size()[0]))), 0)
                all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)
                all_meta_predicted = torch.cat((all_meta_predicted, meta_correct.data.cpu()), 0)

        # Out of distribution data
        for batch_idx, (xs, ys) in enumerate(valloader_noise):
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()
            logits, _ = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            mutual_info = compute_mutual_information(logits)
            all_label = torch.cat((all_label, torch.ones((ys.size()[0]))), 0)
            all_mutual_info = torch.cat((all_mutual_info, mutual_info.data.cpu()), 0)

    # ood Auroc score evaluated using mutual information
    auroc_MI = metrics.roc_auc_score(all_label.numpy(), all_mutual_info.numpy())
    if auroc_MI > best_auroc:
        checkpoint(auroc_MI, epoch)
        best_auroc = auroc_MI

    return

'''
Inference the meta-model on OOD dataset (noisy images)
'''

def OOD(epoch):
    base_net.eval()
    meta_net.eval()

    total_entropy = 0
    max_prob = 0
    mutual_info = 0
    diff_entropy = 0
    precision = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (xs, ys) in enumerate(valloader_noise):
            total += ys.size(0)
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            # test meta model
            logits, _ = compute_logits_and_loss(xs, ys, compute_loss=False)

            # Uncertainty Criterion
            total_entropy += compute_total_entropy(logits).sum()
            max_prob += compute_max_prob(logits).sum()
            mutual_info += compute_mutual_information(logits).sum()
            diff_entropy += compute_differential_entropy(logits).sum()
            precision += compute_precision(logits).sum()

            if args.verbose:
                progress_bar(batch_idx, len(valloader_noise),
                             'DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
                             (diff_entropy/total, mutual_info/total,total_entropy/total,
                              max_prob/total, precision/total))

        if not args.verbose:
            print('DEnt: %.3f | MI: %.3f | TotEnt: %.3f | MaxP: %.3f | Prec: %.3f' %
                  (diff_entropy/total, mutual_info/total, total_entropy/total,
                   max_prob/total, precision/total))

            
def checkpoint(auroc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'meta_net': meta_net.state_dict(),
        'auroc': auroc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/{args.name}_{args.meta_model}_{args.seed}.ckpt')


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    lr /= 10
    if epoch >= 20:
        lr /= 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''
Main training process
'''
if __name__ == '__main__':

    best_loss = float('inf')
    
    time_start = time.perf_counter()
    for epoch in range(start_epoch, args.epoch + 1):
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = validate(epoch)
        if args.name in ['CIFAR10_OOD', 'CIFAR100_OOD', 'MNIST_OOD']:
            OOD(epoch)
            UQ_validation()
            adjust_learning_rate(optimizer, epoch)

        if args.name in ['CIFAR10_miss', 'CIFAR100_miss', 'MNIST_miss', 'SVHN_miss']:

            if valid_loss < best_loss:
              print(f'Validation loss improved from {best_loss:.4e} to {valid_loss:.4e}')  

              checkpoint(0, args.epoch)

              best_loss = valid_loss
            
    print('Finished')
    training_time = time.perf_counter() - time_start
    print('Total training time', str(training_time) + ' seconds')

