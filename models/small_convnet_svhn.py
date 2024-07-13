import torch
import torch.nn as nn
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

  import os
    
  import torchvision.transforms as transforms
  import torchvision.datasets as datasets

  model_pth = os.path.join('../trained-base-models', 'svhn-cnn', 'best.pth')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
  
  model = SmallConvNetSVHN_BaseModel()
  model.load_state_dict(torch.load(model_pth, map_location=device))
  model.eval()

  normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
  ])

  testset = datasets.SVHN(root='~/data/SVHN', split='test', download=True,
                               transform=transform_test)
    
  testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                           num_workers=8)

  n_correct = 0.0
  n_total = 0.0
  for inp, lbl in testloader:
    pred, _ = model(inp)
      
    pred_lbl = torch.argmax(pred, dim=-1)
      
    n_correct += pred_lbl.eq(lbl).sum()
      
    n_total += len(pred)

  print(n_correct / n_total)
