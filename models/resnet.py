##########################################################################################
#
# Taken from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py 
#
##########################################################################################

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    return_nodes = None
    return_nodes_bb = None
    
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(8)
        
        # self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)

        out = self.relu(self.linear1(out))
        out = self.linear2(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32(num_classes):

    model = ResNet(BasicBlock, [5, 5, 5], num_classes) 

    model.return_nodes = {"relu": "x_1", # after 1st conv
                          "layer1.0.relu_1": "x_2",
                          "layer1.1.relu_1": "x_3",
                          "layer1.2.relu_1": 'x_4',
                          "layer1.3.relu_1": 'x_5',
                          "layer1.4.relu_1": "x_6",
                          "layer2.0.relu_1": "x_7",
                          "layer2.1.relu_1": "x_8",
                          "layer2.2.relu_1": "x_9",
                          "layer2.3.relu_1": "x_10",
                          "layer2.4.relu_1": "x_11",
                          "layer3.0.relu_1": "x_12",
                          "layer3.1.relu_1": "x_13",
                          "layer3.2.relu_1": "x_14",
                          "layer3.3.relu_1": "x_15",
                          "layer3.4.relu_1": "x_16",
                          "flatten": "x_17",
                          "linear2": "y_hat"}
    
    model.return_nodes_bb = {"linear2": "y_hat"}

    
    # model.dirichlet_return_nodes = {'conv1': 'fea_1',
    #                                 'layer1.0.relu_1': 'fea2_first',
    #                                 'layer1.4.relu_1': 'fea2_second',
    #                                 'layer2.0.relu_1': 'fea3_first',
    #                                 'layer2.4.relu_1': 'fea3_second',
    #                                 'layer3.0.relu_1': 'fea4_first',
    #                                 'layer3.4.relu_1': 'fea4_second',
    #                                 'flatten': 'fea5'
    #                                 }
    
    model.dirichlet_return_nodes = {'layer2.0.relu_1': 'fea3_first',
                                    'layer2.4.relu_1': 'fea3_second',
                                    'layer3.0.relu_1': 'fea4_first',
                                    'layer3.4.relu_1': 'fea4_second',
                                    'flatten': 'fea5',
                                    'linear2': 'out'}
    
    # print('FIXME: using different return nodes')
    
    # model.return_nodes = {"layer2.3.relu_1": "layer2.3.relu_1",
    #                       "layer2.4.relu": "layer2.4.relu",
    #                       "layer2.4.relu_1": "layer2.4.relu_1",
    #                       "layer3.0.relu": "layer3.0.relu",
    #                       "layer3.0.relu_1": "layer3.0.relu_1",
    #                       "layer3.1.relu": "layer3.1.relu",
    #                       "layer3.1.relu_1": "layer3.1.relu_1",
    #                       "layer3.2.relu": "layer3.2.relu",
    #                       "layer3.2.relu_1": "layer3.2.relu_1",
    #                       "layer3.3.relu": "layer3.3.relu",
    #                       "layer3.3.relu_1": "layer3.3.relu_1",
    #                       "layer3.4.relu": "layer3.4.relu",
    #                       "layer3.4.relu_1": "layer3.4.relu_1",
    #                       "flatten": "flatten",
    #                       "linear1": "linear1",
    #                       "linear2": "linear2"}
    
    # model.return_nodes = {"layer2.3.relu_1": "layer2.3.relu_1",
    #                       "layer3.1.relu": "layer3.1.relu",
    #                       }

    
    # model.return_nodes = {
    #                       "layer3.1.relu": "layer3.1.relu"
    #                       }
    
    return model


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

class ResNetWrapper(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
      
    resnet = resnet32(num_classes)
    self.fe = create_feature_extractor(resnet, resnet.dirichlet_return_nodes)
            
  def forward(self, x):
    fe_res = {k: v.flatten(start_dim=1) for k, v in self.fe(x).items()}

    out = fe_res.pop('out')

    return out, list(fe_res.values())

  # re-define load_state_dict so pytorch does not complain about names
  def load_state_dict(self, state_dict):
      self.fe.load_state_dict(state_dict)


class Resnet_MetaModel_combine(nn.Module):
  def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4, fea_dim5, n_classes):
    super().__init__()

    self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
    
    self.classifier3_fc1 = nn.Linear(4096, 2048)
    self.classifier3_fc2 = nn.Linear(1024, 512)
    self.classifier3_fc3 = nn.Linear(256, 128)
    self.classifier3_fc4 = nn.Linear(64, n_classes)

    self.classifier3_fc1_second = nn.Linear(4096, 2048)
    self.classifier3_fc2_second = nn.Linear(1024, 512)
    self.classifier3_fc3_second = nn.Linear(256, 128)
    self.classifier3_fc4_second = nn.Linear(64, n_classes)
    
    self.classifier4_fc1 = nn.Linear(2048, 1024)
    self.classifier4_fc2 = nn.Linear(512, 256)
    self.classifier4_fc3 = nn.Linear(128, 64)
    self.classifier4_fc4 = nn.Linear(64, n_classes)

    self.classifier4_fc1_second = nn.Linear(2048, 1024)
    self.classifier4_fc2_second = nn.Linear(512, 256)
    self.classifier4_fc3_second = nn.Linear(128, 64)
    self.classifier4_fc4_second = nn.Linear(64, n_classes)

    self.classifier5_fc1 = nn.Linear(64, n_classes)

    self.classifier_final = nn.Linear(5*n_classes, n_classes)

  def forward(self, fea3, fea3_second, fea4, fea4_second, fea5):
      
    fea3 = self.pooling(fea3)
    fea3 = F.relu(self.classifier3_fc1(fea3))
    fea3 = self.pooling(fea3)
    fea3 = F.relu(self.classifier3_fc2(fea3))
    fea3 = self.pooling(fea3)
    fea3 = F.relu(self.classifier3_fc3(fea3))
    fea3 = self.pooling(fea3)
    fea3 = F.relu(self.classifier3_fc4(fea3))

    fea3_second = self.pooling(fea3_second)
    fea3_second = F.relu(self.classifier3_fc1_second(fea3_second))
    fea3_second = self.pooling(fea3_second)
    fea3_second = F.relu(self.classifier3_fc2_second(fea3_second))
    fea3_second = self.pooling(fea3_second)
    fea3_second = F.relu(self.classifier3_fc3_second(fea3_second))
    fea3_second = self.pooling(fea3_second)
    fea3_second = F.relu(self.classifier3_fc4_second(fea3_second))

    fea4 = self.pooling(fea4)
    fea4 = F.relu(self.classifier4_fc1(fea4))
    fea4 = self.pooling(fea4)
    fea4 = F.relu(self.classifier4_fc2(fea4))
    fea4 = self.pooling(fea4)
    fea4 = F.relu(self.classifier4_fc3(fea4))
    fea4 = F.relu(self.classifier4_fc4(fea4))

    fea4_second = self.pooling(fea4_second)
    fea4_second = F.relu(self.classifier4_fc1_second(fea4_second))
    fea4_second = self.pooling(fea4_second)
    fea4_second = F.relu(self.classifier4_fc2_second(fea4_second))
    fea4_second = self.pooling(fea4_second)
    fea4_second = F.relu(self.classifier4_fc3_second(fea4_second))
    fea4_second = F.relu(self.classifier4_fc4_second(fea4_second))

    fea5 = F.relu(self.classifier5_fc1(fea5))

    fea = torch.cat((fea3, fea3_second, fea4, fea4_second, fea5), 1)
    z = self.classifier_final(fea)

    return z

    
def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":

    import os
    
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    model_pth = os.path.join('../trained-base-models', 'cifar10-resnet32', 'best.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    wrapper = ResNetWrapper(10).to(device)
    wrapper.load_state_dict(torch.load(model_pth, map_location=device))
    wrapper.eval()
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

    # for name, param in wrapper.named_parameters():
    #     print(name, param.shape)
    
    testset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=True,
                               transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    n_correct = 0.0
    n_total = 0.0
    for inp, lbl in testloader:
      pred, _ = wrapper(inp)
      
      pred_lbl = torch.argmax(pred, dim=-1)
      
      n_correct += pred_lbl.eq(lbl).sum()
      
      n_total += len(pred)

    print(n_correct / n_total)
    
