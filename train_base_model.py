from __future__ import print_function
import argparse
import csv
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import random

import models
from utils import progress_bar, set_seed, convert_to_rgb
from model_utils import * 

parser = argparse.ArgumentParser(description='Base model training')
# parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="VGG16_BaseModel", type=str, help='model type (default: VGG16_BaseModel)')
parser.add_argument('--name', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=100, type=int, help='total epochs to run')
# parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
# parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--verbose', action='store_true', help='Weather or not to use progress bar during model training (default: False).')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

set_seed(args.seed, use_cuda)

args.batch_size = 128

if args.dataset == 'CIFAR10':
    args.epoch = 250    
elif args.dataset == 'CIFAR100':
    args.epoch = 170
else:
    args.epoch = 100
    
'''
Processing data
'''
print('==> Preparing data..')
if args.dataset=='CIFAR10':

    transform_train, _ = get_preprocessor(args.dataset) 
    
    dataset = datasets.CIFAR10(root='~/data', train=True, download=True,
                               transform=transform_train)

    train_idxs = np.load('dataset_idxs/cifar10/train_idx.npy')
    val_idxs = np.load('dataset_idxs/cifar10/val_idx.npy')
    
    train_dataset = data.Subset(dataset, train_idxs)
    val_dataset = data.Subset(dataset, val_idxs)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    
elif args.dataset=='CIFAR100':

    transform_train, _ = get_preprocessor(args.dataset) 

    dataset = datasets.CIFAR100(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    train_idxs = np.load('dataset_idxs/cifar100/train_idx.npy')
    val_idxs = np.load('dataset_idxs/cifar100/val_idx.npy')
    
    train_dataset = data.Subset(dataset, train_idxs)
    val_dataset = data.Subset(dataset, val_idxs)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)

elif args.dataset == 'MNIST':

    transform_train, _ = get_preprocessor(args.dataset) 
    
    dataset = datasets.MNIST(root='~/data', train=True, download=True, transform=transform_train)

    train_idxs = np.load('dataset_idxs/mnist/train_idx.npy')
    val_idxs = np.load('dataset_idxs/mnist/val_idx.npy')
    
    train_dataset = data.Subset(dataset, train_idxs)
    val_dataset = data.Subset(dataset, val_idxs)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    
elif args.dataset == 'SVHN':

    transform_train, _ = get_preprocessor(args.dataset)

    dataset = datasets.SVHN(root='~/data', split='train', download=True, transform=transform_train)
    
    train_idxs = np.load('dataset_idxs/svhn/train_idx.npy')
    val_idxs = np.load('dataset_idxs/svhn/val_idx.npy')
    
    train_dataset = data.Subset(dataset, train_idxs)
    val_dataset = data.Subset(dataset, val_idxs)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)

'''
Preparing model
'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/ckpt.t7{args.name}_{args.seed}_{args.epoch}')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    if args.dataset == 'CIFAR100':
        net = models.__dict__[args.model](100)
        net.init_vgg16_params()
    elif args.dataset == 'CIFAR10':
        net = models.__dict__[args.model](10)
        net.init_vgg16_params()
    else:
        net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')

logname = f'results/log_{net.__class__.__name__}_{args.name}_{args.seed}.csv'

if use_cuda:
    net = net.cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

print("Training model on device:", next(net.parameters()).device)

criterion = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss()

if args.dataset == 'CIFAR10':
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
elif args.dataset == 'CIFAR100':
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], verbose=True)

else: # MNIST, SVHN
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    
'''
Training model
'''
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.verbose:
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'%
                         (train_loss/(batch_idx+1), 100*correct/total, correct, total))

    if not args.verbose:
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'%
              (train_loss/len(trainloader), 100*correct/total, correct, total))

    return (train_loss/batch_idx, 100*correct/total)

'''
Testing model
'''
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            loss = criterion_test(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if args.verbose:
                progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                
    if not args.verbose:
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / len(valloader), 100. * correct / total, correct, total))

    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc

    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, f'./checkpoint/ckpt.t7{args.name}_{args.seed}_{args.epoch}')

# setup log file        
with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc', 'val loss', 'val acc'])

# Train model
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    if args.dataset == 'CIFAR100':
        scheduler.step()
    
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

