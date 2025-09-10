# neural net stuff for material classification
# based on resnet but simplified

import torch
import torch.nn as nn

class MaterialNet(nn.Module):
    # CNN for classifying materials - 6 types
    # Learned from Machine Learning for Language and Vision with Professor Nathan


    def __init__(self, num_classes=6):
        super().__init__()
        
        # first conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # residual blocks - copied from some tutorial
        self.block1 = self.make_block(64, 64, 2)
        self.block2 = self.make_block(64, 128, 2, stride=2)
        self.block3 = self.make_block(128, 256, 2, stride=2)
        self.block4 = self.make_block(256, 512, 2, stride=2)
        
        # final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def make_block(self, in_ch, out_ch, num_blocks, stride=1):
        # helper to make residual blocks
        layers = []
        layers.append(ResBlock(in_ch, out_ch, stride))
        for i in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResBlock(nn.Module):
    # basic residual block
    
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)  # skip connection
        out = self.relu(out)
        
        return out

def count_params(model):
    # count how many parameters the model has
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 * 1024)  # assume float32
    
    return {
        'total': total,
        'trainable': trainable,
        'size_mb': size_mb
    }

def save_model(model, path, extra_stuff=None):
    # save model to file
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'state_dict': model.state_dict(),
        'model_type': 'MaterialNet',
        'classes': 6
    }
    
    if extra_stuff:
        save_dict.update(extra_stuff)
    
    torch.save(save_dict, path)
    # Removed print statement to avoid duplicate output

def load_model(path):
    # load model from file
    checkpoint = torch.load(path, map_location='cpu')
    
    model = MaterialNet(num_classes=6)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model
