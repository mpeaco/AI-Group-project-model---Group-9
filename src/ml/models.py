"""
Neural network models for material recognition
CNN architectures for the laser cutting project
"""

import torch
import torch.nn as nn

class MaterialNet(nn.Module):
    """Advanced CNN for material classification with residual connections"""
    
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Initial convolution
        self.conv_initial = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn_initial = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 64, 2)
        self.res_block2 = self._make_residual_block(64, 128, 2, stride=2)
        self.res_block3 = self._make_residual_block(128, 256, 2, stride=2)
        self.res_block4 = self._make_residual_block(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def _make_residual_block(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a residual block"""
        layers = []
        
        # First block might have stride > 1
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Additional blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial processing
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for MaterialNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut connection
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

def get_model_info(model):
    """Get information about a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'type': type(model).__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def save_model(model, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        'num_classes': 6
    }
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(path):
    checkpoint = torch.load(path, map_location='cpu')
    
    model = MaterialNet(num_classes=6)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def compare_models():
    print("MODEL COMPARISON")
    print("=" * 40)
    
    model = MaterialNet(num_classes=6)
    info = get_model_info(model)
    print(f"\nMaterialNet:")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.1f} MB")

def test_model_forward():
    print("\nMODEL FORWARD PASS TEST")
    print("=" * 40)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    model = MaterialNet(num_classes=6)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"MaterialNet: Input {list(dummy_input.shape)} -> Output {list(output.shape)}")
