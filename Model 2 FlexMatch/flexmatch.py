import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
import logging

class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class FlexMatch:
    def __init__(self, num_classes=10, device='cuda', lambda_u=1.0, T=1.0, threshold=0.95):
        self.device = device
        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.T = T
        self.threshold = threshold
        
        # Create model
        self.model = WideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0)
        self.ema_model = WideResNet(num_classes=num_classes, depth=28, widen_factor=2, dropRate=0.0)
        self.model.to(device)
        self.ema_model.to(device)
        
        # Initialize EMA model with same weights
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1024)
        
        # FlexMatch specific variables
        self.class_threshold = torch.ones(num_classes, device=device) * threshold
        self.class_counts = torch.zeros(num_classes, device=device)
        
    def train_step(self, labeled_batch, unlabeled_batch_weak, unlabeled_batch_strong):
        x_l, y_l = labeled_batch
        x_uw, _ = unlabeled_batch_weak
        x_us, _ = unlabeled_batch_strong
        
        x_l = x_l.to(self.device)
        y_l = y_l.to(self.device)
        x_uw = x_uw.to(self.device)
        x_us = x_us.to(self.device)
        
        batch_size = x_l.size(0)
        
        # Forward pass on labeled data
        logits_l = self.model(x_l)
        loss_l = F.cross_entropy(logits_l, y_l)
        
        # Forward pass on unlabeled data
        with torch.no_grad():
            logits_uw = self.model(x_uw)
            probs_uw = torch.softmax(logits_uw.detach() / self.T, dim=-1)
            max_probs, pseudo_labels = torch.max(probs_uw, dim=-1)
        
        # Forward pass on strongly augmented unlabeled data
        logits_us = self.model(x_us)
        probs_us = torch.softmax(logits_us, dim=-1)
        
        # FlexMatch threshold adjustment
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                class_mask = (pseudo_labels == class_idx)
                if class_mask.sum() > 0:
                    self.class_counts[class_idx] = class_mask.sum()
                    self.class_threshold[class_idx] = (self.threshold * 
                                                     (self.class_counts[class_idx] / 
                                                      (self.class_counts.max() + 1e-12)))
        
        # Mask for confident predictions
        mask = max_probs.ge(self.class_threshold[pseudo_labels])
        
        if mask.sum() > 0:
            loss_u = (F.cross_entropy(logits_us, pseudo_labels, reduction='none') * mask).mean()
        else:
            loss_u = torch.tensor(0.0, device=self.device)
        
        total_loss = loss_l + self.lambda_u * loss_u
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update EMA model
        self.update_ema()
        
        return {
            'loss_l': loss_l.item(),
            'loss_u': loss_u.item() if mask.sum() > 0 else 0.0,
            'total_loss': total_loss.item(),
            'mask_ratio': mask.float().mean().item()
        }
    
    def update_ema(self):
        alpha = 0.999
        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)
    
    def predict(self, x):
        self.ema_model.eval()
        with torch.no_grad():
            logits = self.ema_model(x.to(self.device))
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu()
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])