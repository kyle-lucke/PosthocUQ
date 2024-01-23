import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Original implementation from ConfidNet:
# https://github.com/valeoai/ConfidNet/blob/master/confidnet/models/small_convnet_svhn.py 
class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)

'''
VGG16 base-model for CIFAR10
'''

##########
### OLD
##########

# class VGG16_BaseModel(nn.Module):
#     def __init__(self):
#         super(VGG16_BaseModel, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
#         self.batchnorm1 = nn.BatchNorm2d(64)

#         self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
#         self.batchnorm2 = nn.BatchNorm2d(128)

#         self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.batchnorm3 = nn.BatchNorm2d(256)

#         self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.batchnorm4 = nn.BatchNorm2d(512)

#         self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.batchnorm5 = nn.BatchNorm2d(512)

#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 10)
#         self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = F.relu(self.conv1_1(x))
#         x = F.relu(self.batchnorm1(self.conv1_2(x)))
#         x = self.pooling(x)
#         fea1 = x.view(x.size(0), -1)
#         x = F.relu(self.conv2_1(x))
#         x = F.relu(self.batchnorm2(self.conv2_2(x)))
#         x = self.pooling(x)
#         fea2 = x.view(x.size(0), -1)
#         x = F.relu(self.conv3_1(x))
#         x = F.relu(self.conv3_2(x))
#         x = F.relu(self.batchnorm3(self.conv3_3(x)))
#         x = self.pooling(x)
#         fea3 = x.view(x.size(0), -1)
#         x = F.relu(self.conv4_1(x))
#         x = F.relu(self.conv4_2(x))
#         x = F.relu(self.batchnorm4(self.conv4_3(x)))
#         x = self.pooling(x)
#         fea4 = x.view(x.size(0), -1)
#         x = F.relu(self.conv5_1(x))
#         x = F.relu(self.conv5_2(x))
#         x = F.relu(self.batchnorm5(self.conv5_3(x)))
#         x = self.pooling(x)
#         fea5 = x.view(x.shape[0], -1)

#         feature = F.relu(self.fc1(fea5))
#         predict = self.fc2(feature)
#         return predict, [fea1, fea2, fea3, fea4, fea5]

class VGG16_BaseModel(nn.Module):

  def __init__(self, n_classes):
    super().__init__()

    self.conv1 = Conv2dSame(3, 64, 3)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv1_dropout = nn.Dropout(0.3)
    self.conv2 = Conv2dSame(64, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)
    self.maxpool1 = nn.MaxPool2d(2)
    
    self.conv3 = Conv2dSame(64, 128, 3)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv3_dropout = nn.Dropout(0.4)
    self.conv4 = Conv2dSame(128, 128, 3)
    self.bn4 = nn.BatchNorm2d(128)
    self.maxpool2 = nn.MaxPool2d(2)
    
    self.conv5 = Conv2dSame(128, 256, 3)
    self.bn5 = nn.BatchNorm2d(256)
    self.conv5_dropout = nn.Dropout(0.4)
    self.conv6 = Conv2dSame(256, 256, 3)
    self.bn6 = nn.BatchNorm2d(256)
    self.conv6_dropout = nn.Dropout(0.4)
    self.conv7 = Conv2dSame(256, 256, 3)
    self.bn7 = nn.BatchNorm2d(256)
    self.maxpool3 = nn.MaxPool2d(2)

    self.conv8 = Conv2dSame(256, 512, 3)
    self.bn8 = nn.BatchNorm2d(512)
    self.conv8_dropout = nn.Dropout(0.4)
    self.conv9 = Conv2dSame(512, 512, 3)
    self.bn9 = nn.BatchNorm2d(512)
    self.conv9_dropout = nn.Dropout(0.4)
    self.conv10 = Conv2dSame(512, 512, 3)
    self.bn10 = nn.BatchNorm2d(512)
    self.maxpool4 = nn.MaxPool2d(2)

    self.conv11 = Conv2dSame(512, 512, 3)
    self.bn11 = nn.BatchNorm2d(512)
    self.conv11_dropout = nn.Dropout(0.4)
    self.conv12 = Conv2dSame(512, 512, 3)
    self.bn12 = nn.BatchNorm2d(512)
    self.conv12_dropout = nn.Dropout(0.4)
    self.conv13 = Conv2dSame(512, 512, 3)
    self.bn13 = nn.BatchNorm2d(512)
    self.maxpool5 = nn.MaxPool2d(2)

    self.end_dropout = nn.Dropout(0.5)

    self.fc1 = nn.Linear(512, 512)
    self.dropout_fc = nn.Dropout(0.5)
    self.fc2 = nn.Linear(512, n_classes)

    self.relu = nn.ReLU()
    
  def forward(self, x):

    out = self.relu(self.conv1(x))
    out = self.bn1(out)
    out = self.conv1_dropout(out)
    out = self.relu(self.conv2(out))
    out = self.bn2(out)
    out = self.maxpool1(out)

    fea1 = out.view(out.size(0), -1)
    
    out = self.relu(self.conv3(out))
    out = self.bn3(out)
    out = self.conv3_dropout(out)
    out = self.relu(self.conv4(out))
    out = self.bn4(out)
    out = self.maxpool2(out)

    fea2 = out.view(out.size(0), -1)
    
    out = self.relu(self.conv5(out))
    out = self.bn5(out)
    out = self.conv5_dropout(out)
    out = self.relu(self.conv6(out))
    out = self.bn6(out)
    out = self.conv6_dropout(out)
    out = self.relu(self.conv7(out))
    out = self.bn7(out)
    out = self.maxpool3(out)

    fea3 = out.view(out.size(0), -1)
    
    out = self.relu(self.conv8(out))
    out = self.bn8(out)
    out = self.conv8_dropout(out)
    out = self.relu(self.conv9(out))
    out = self.bn9(out)
    out = self.conv9_dropout(out)
    out = self.relu(self.conv10(out))
    out = self.bn10(out)
    out = self.maxpool4(out)

    fea4 = out.view(out.size(0), -1)
    
    out = self.relu(self.conv11(out))
    out = self.bn11(out)
    out = self.conv11_dropout(out)
    out = self.relu(self.conv11(out))
    out = self.bn11(out)
    out = self.conv11_dropout(out)
    out = self.relu(self.conv12(out))
    out = self.bn12(out)
    out = self.conv12_dropout(out)
    out = self.relu(self.conv13(out))
    out = self.bn13(out)
    out = self.maxpool5(out)

    fea5 = out.view(out.size(0), -1)
    
    out = self.end_dropout(out)
    out = torch.flatten(out, start_dim=1)
    out = self.relu(self.fc1(out))

    out = self.dropout_fc(out)
    out = self.fc2(out)

    return out, [fea1, fea2, fea3, fea4, fea5]

  def init_vgg16_params(self):
    vgg16 = models.vgg16(pretrained=True)
    vgg_layers = []
    for _layer in vgg16.features.children():
      if isinstance(_layer, nn.Conv2d):
        vgg_layers.append(_layer)

    model_layers = [
      self.conv1,
      self.conv2,
      self.conv3,
      self.conv4,
      self.conv5,
      self.conv6,
      self.conv7,
      self.conv8,
      self.conv9,
      self.conv10,
      self.conv11,
      self.conv12,
      self.conv13,
    ]

    assert len(vgg_layers) == len(model_layers)

    for l1, l2 in zip(vgg_layers, model_layers):
      if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        assert l1.weight.size() == l2.weight.size()
        assert l1.bias.size() == l2.bias.size()
        l2.weight.data = l1.weight.data
        l2.bias.data = l1.bias.data

'''
VGG16 meta-model
'''

class VGG16_MetaModel_combine(nn.Module):
    def __init__(self, fea_dim1, fea_dim2, fea_dim3, fea_dim4, fea_dim5, n_classes):
        super(VGG16_MetaModel_combine, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.classifier1_fc1 = nn.Linear(fea_dim1, 8192)
        self.classifier1_fc2 = nn.Linear(4096, 2048)
        self.classifier1_fc3 = nn.Linear(1024, 512)
        self.classifier1_fc4 = nn.Linear(256, n_classes)

        self.classifier2_fc1 = nn.Linear(fea_dim2, 4096)
        self.classifier2_fc2 = nn.Linear(2048, 1024)
        self.classifier2_fc3 = nn.Linear(512, 256)
        self.classifier2_fc4 = nn.Linear(256, n_classes)

        self.classifier3_fc1 = nn.Linear(fea_dim3, 2048)
        self.classifier3_fc2 = nn.Linear(1024, 512)
        self.classifier3_fc3 = nn.Linear(512, 256)
        self.classifier3_fc4 = nn.Linear(256, n_classes)

        self.classifier4_fc1 = nn.Linear(fea_dim4, 1024)
        self.classifier4_fc2 = nn.Linear(512, 256)
        self.classifier4_fc3 = nn.Linear(256, n_classes)

        self.classifier5_fc1 = nn.Linear(fea_dim5, 256)
        self.classifier5_fc2 = nn.Linear(256, n_classes)

        self.classifier_final = nn.Linear(5 * n_classes, n_classes)

    def forward(self, fea1, fea2, fea3, fea4, fea5):
        fea1 = F.relu(self.classifier1_fc1(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc2(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc3(fea1))
        fea1 = self.pooling(fea1)
        fea1 = F.relu(self.classifier1_fc4(fea1))

        fea2 = F.relu(self.classifier2_fc1(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc2(fea2))
        fea2 = self.pooling(fea2)
        fea2 = F.relu(self.classifier2_fc3(fea2))
        fea2 = F.relu(self.classifier2_fc4(fea2))

        fea3 = F.relu(self.classifier3_fc1(fea3))
        fea3 = self.pooling(fea3)
        fea3 = F.relu(self.classifier3_fc2(fea3))
        fea3 = F.relu(self.classifier3_fc3(fea3))
        fea3 = F.relu(self.classifier3_fc4(fea3))

        fea4 = F.relu(self.classifier4_fc1(fea4))
        fea4 = self.pooling(fea4)
        fea4 = F.relu(self.classifier4_fc2(fea4))
        fea4 = F.relu(self.classifier4_fc3(fea4))

        fea5 = F.relu(self.classifier5_fc1(fea5))
        fea5 = F.relu(self.classifier5_fc2(fea5))

        fea = torch.cat((fea1, fea2, fea3, fea4, fea5), 1)
        z = self.classifier_final(fea)
        return z

if __name__ == '__main__':

  torch.manual_seed(0)
  r_data = torch.rand(128, 3, 32, 32)

  model = VGG16_BaseModel(10)
  model.eval()
  
  res1, ftrs = model(r_data)
  res2, ftrs = model(r_data)

  for f in ftrs:
    print(f.shape)
    
  print(res2.shape)
  print()
  
  print(torch.mean(res1 - res2))
  print()

  
