import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallConvNetMNIST_BaseModel(nn.Module):
  
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 32, 3)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.maxpool = nn.MaxPool2d(2)
    self.dropout1 = nn.Dropout(0.25)

    self.fc1 = nn.Linear(9216, 128)
    self.dropout2 = nn.Dropout(0.5)

    self.fc2 = nn.Linear(128, 10)

    self.relu = nn.ReLU()
    
  def forward(self, x):
    out = self.relu(self.conv1(x))

    fea1 = F.max_pool2d(out, 2)
    fea1 = fea1.view(fea1.size(0), -1)
    
    out = self.relu(self.conv2(out))
    out = self.maxpool(out)

    fea2 = out.view(out.size(0), -1)
    
    out = self.dropout1(out)

    out = torch.flatten(out, start_dim=1)
    
    out = self.relu(self.fc1(out))
    out = self.dropout2(out)

    out = self.fc2(out)

    return out, [fea1, fea2]

# FIXME: figure out meta model architechure
class SmallConvNetMNIST_MetaModel_combine(nn.Module):
  def __init__(self, fea_dim1, fea_dim2):
    super().__init__()

    self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
    
    # Meta Model Layers
    self.classifier1_fc1 = nn.Linear(fea_dim3, 2048)
    self.classifier1_fc2 = nn.Linear(1024, 512)
    self.classifier1_fc3 = nn.Linear(512, 256)
    self.classifier1_fc4 = nn.Linear(256, 10)

    self.classifier2_fc1 = nn.Linear(fea_dim2, 4096)
    self.classifier2_fc2 = nn.Linear(2048, 1024)
    self.classifier2_fc3 = nn.Linear(512, 256)
    self.classifier2_fc4 = nn.Linear(256, 10)

    self.classifier_final = nn.Linear(10 * 2, 10)

  def forward(self, x, y):
    # Meta Model

    x = F.relu(self.classifier3_fc1(x))
    x = self.pooling(x)
    x = F.relu(self.classifier3_fc2(x))
    x = F.relu(self.classifier3_fc3(x))
    x = F.relu(self.classifier3_fc4(x))

    y = F.relu(self.classifier2_fc1(y))
    y = self.pooling(y)
    y = F.relu(self.classifier2_fc2(y))
    y = self.pooling(y)
    y = F.relu(self.classifier2_fc3(y))
    y = F.relu(self.classifier2_fc4(y))

    z = torch.cat((x, y), 1)

    z = self.classifier_final(z)

    return z

if __name__ == '__main__':

  torch.manual_seed(0)
  r_data = torch.rand(128, 1, 28, 28)

  model = SmallConvNetMNIST_BaseModel()
  model.eval()
  
  res1, ftrs = model(r_data)
  res2, ftrs = model(r_data)

  for f in ftrs:
    print(f.shape)
    
  # print(res2.shape)
  # print()
  
  # print(torch.mean(res1 - res2))
  # print()
