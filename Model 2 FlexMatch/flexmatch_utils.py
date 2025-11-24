import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert numpy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

class CIFAR100Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert numpy array to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

def get_cifar100_dataset():
    """Load CIFAR-100 dataset for pre-training"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def get_cifar10_datasets(labeled_ratio=0.05):
    """Load CIFAR-10 dataset and split into labeled and unlabeled sets"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_strong = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split into labeled and unlabeled
    data = train_dataset.data
    targets = train_dataset.targets
    
    # Create labeled and unlabeled splits
    labeled_data = []
    labeled_targets = []
    unlabeled_data = []
    unlabeled_targets = []
    
    num_classes = 10
    labeled_per_class = int(len(data) * labeled_ratio / num_classes)
    
    for class_idx in range(num_classes):
        class_indices = np.where(np.array(targets) == class_idx)[0]
        np.random.shuffle(class_indices)
        
        labeled_indices = class_indices[:labeled_per_class]
        unlabeled_indices = class_indices[labeled_per_class:]
        
        labeled_data.extend(data[labeled_indices])
        labeled_targets.extend([targets[i] for i in labeled_indices])
        unlabeled_data.extend(data[unlabeled_indices])
        unlabeled_targets.extend([targets[i] for i in unlabeled_indices])
    
    # Convert to numpy arrays for easier handling
    labeled_data = np.array(labeled_data)
    unlabeled_data = np.array(unlabeled_data)
    
    # Create datasets
    labeled_dataset = CIFAR10Dataset(labeled_data, labeled_targets, transform=transform_train)
    unlabeled_dataset = CIFAR10Dataset(unlabeled_data, unlabeled_targets, transform=transform_train)
    unlabeled_dataset_strong = CIFAR10Dataset(unlabeled_data, unlabeled_targets, transform=transform_strong)
    
    return labeled_dataset, unlabeled_dataset, unlabeled_dataset_strong, test_dataset

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)