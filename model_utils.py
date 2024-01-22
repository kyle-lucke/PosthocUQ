import torch
import torchvision.transforms as transforms

def get_preprocessor(dataset):
  if model_name == 'MNIST':

    normalize = transforms.Normalize(mean=[0.1307,],
                                 std=[0.3081,])

    transform_train = transforms.Compose([
      transforms.ToTensor(),
      normalize])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
    
  elif model_name == 'SVHN':

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose([
      transforms.ToTensor(),
      normalize])
    
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

    
  elif model_name == 'CIFAR10':

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),    
      transforms.ToTensor(),
      normalize])

    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
    
  elif model_name == 'CIFAR100':

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

    transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      normalize])

    transform_test = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

  else:
    raise Exception('ERROR: unrecognized model name.')
    
  return transform_train, transform_test
