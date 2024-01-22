import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import *

class SmallConvNetSVHN_BaseModel(nn.Module):
  
  def __init__(self):
    super().__init__()

    feature_dim = 512

    self.conv1 = Conv2dSame(3, 32, 3)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = Conv2dSame(32, 32, 3)
    self.bn2 = nn.BatchNorm2d(32)
    self.maxpool1 = nn.MaxPool2d(2)
    self.dropout1 = nn.Dropout(0.3)

    self.conv3 = Conv2dSame(32, 64, 3)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = Conv2dSame(64, 64, 3)
    self.bn4 = nn.BatchNorm2d(64)
    self.maxpool2 = nn.MaxPool2d(2)
    self.dropout2 = nn.Dropout(0.3)

    self.conv5 = Conv2dSame(64, 128, 3)
    self.bn5 = nn.BatchNorm2d(128)
    self.conv6 = Conv2dSame(128, 128, 3)
    self.bn6 = nn.BatchNorm2d(128)
    self.maxpool3 = nn.MaxPool2d(2)
    self.dropout3 = nn.Dropout(0.3)

    self.fc1 = nn.Linear(2048, feature_dim)
    self.dropout4 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(feature_dim, 10)

    self.relu = nn.ReLU()
    
  def forward(self, x):

    out = self.relu(self.conv1(x))
    out = self.bn1(out)
    out = self.relu(self.conv2(out))
    out = self.bn2(out)
    out = self.maxpool1(out)

    fea1 = out.view(out.size(0), -1)
    
    out = self.dropout1(out)

    out = self.relu(self.conv3(out))
    out = self.bn3(out)
    out = self.relu(self.conv4(out))
    out = self.bn4(out)
    out = self.maxpool2(out)

    fea2 = out.view(out.size(0), -1)
    
    out = self.dropout2(out)

    out = self.relu(self.conv5(out))
    out = self.bn5(out)
    out = self.relu(self.conv6(out))
    out = self.bn6(out)
    out = self.maxpool3(out)

    fea3 = out.view(out.size(0), -1)
    
    out = self.dropout3(out)

    out = torch.flatten(out, start_dim=1)
    out = self.relu(self.fc1(out))
    out = self.dropout4(out)

    out = self.fc2(out)
    return out, [fea1, fea2, fea3]

class SmallConvNetSVHN_MetaModel_combine(nn.Module):
  def __init__(self, fea_dim1, fea_dim2, fea_dim3):
    super().__init__()
    # Meta Model Layers

    self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
    
    self.classifier1_fc1 = nn.Linear(fea_dim1, 4096)
    self.classifier1_fc2 = nn.Linear(2048, 1024)
    self.classifier1_fc3 = nn.Linear(512, 256)
    self.classifier1_fc4 = nn.Linear(256, 10)

    self.classifier2_fc1 = nn.Linear(fea_dim2, 2048)
    self.classifier2_fc2 = nn.Linear(1024, 512)
    self.classifier2_fc3 = nn.Linear(512, 256)
    self.classifier2_fc4 = nn.Linear(256, 10)

    self.classifier3_fc1 = nn.Linear(fea_dim3, 1024)
    self.classifier3_fc2 = nn.Linear(512, 256)
    self.classifier3_fc3 = nn.Linear(256, 10)

    self.classifier_final = nn.Linear(10 * 3, 10)

  def forward(self, fea1, fea2, fea3):
    # Meta Model
    fea1 = F.relu(self.classifier1_fc1(fea1))
    fea1 = self.pooling(fea1)
    fea1 = F.relu(self.classifier1_fc2(fea1))
    fea1 = self.pooling(fea1)
    fea1 = F.relu(self.classifier1_fc3(fea1))
    fea1 = F.relu(self.classifier1_fc4(fea1))

    fea2 = F.relu(self.classifier2_fc1(fea2))
    fea2 = self.pooling(fea2)
    fea2 = F.relu(self.classifier2_fc2(fea2))
    fea2 = F.relu(self.classifier2_fc3(fea2))
    fea2 = F.relu(self.classifier2_fc4(fea2))

    fea3 = F.relu(self.classifier3_fc1(fea3))
    fea3 = self.pooling(fea3)
    fea3 = F.relu(self.classifier3_fc2(fea3))
    fea3 = F.relu(self.classifier3_fc3(fea3))

    z = torch.cat((fea1, fea2, fea3), 1)
    
    z = self.classifier_final(z)

    return z
  
if __name__ == '__main__':

  torch.manual_seed(0)
  r_data = torch.rand(128, 3, 32, 32)

  model = SmallConvNetSVHN_BaseModel()
  model.eval()
  
  res1, ftrs = model(r_data)
  res2, ftrs = model(r_data)
  
  for f in ftrs:
    print(f.shape)
  print()
    
  f1, f2, f3 = [f.shape[1] for f in ftrs]
    
  mm = SmallConvNetSVHN_MetaModel_combine(f1, f2, f3)
  mm_res = mm(*ftrs)

  print(mm_res.shape)
  
  print(torch.mean(res1 - res2))
  print()
