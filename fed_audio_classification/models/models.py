from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchModelOptimized(nn.Module):
    def __init__(self, input_shape=(160, 345, 2), num_classes=10, seed=42):
        super().__init__()
        
        torch.manual_seed(seed)
        in_channels = input_shape[2]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),  # Kernel standardizzato
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),  # Kernel standardizzato
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling integrato
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),  
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Linear):
               
                if module.out_features == self.classifier[-1].out_features:
                    
                    nn.init.xavier_normal_(module.weight)
                else:
                    
                    nn.init.kaiming_normal_(
                        module.weight, 
                        mode='fan_in', 
                        nonlinearity='relu'
                    )
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classificatore
        x = self.classifier(x)
        
        return x


def get_parameters(net):
    """Extract model parameters"""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    """Set model parameters (including buffers)."""
    keys = list(net.state_dict().keys())

    if len(keys) != len(parameters):
        raise ValueError(
            f"Mismatch in parameters: expected {len(keys)}, received {len(parameters)}"
        )

    # Debug: print shape mismatch if any
    for k, p in zip(keys, parameters):
        expected_shape = net.state_dict()[k].shape
        if tuple(p.shape) != tuple(expected_shape):
            print(f"[Mismatch] {k}: expected {expected_shape}, got {p.shape}")

    # Rebuild state_dict
    state_dict = OrderedDict({
        k: torch.tensor(p) for k, p in zip(keys, parameters)
    })

    net.load_state_dict(state_dict, strict=True)
    

def create_model_with_fixed_seed(input_shape=(160, 345, 2), num_classes=10, seed=42):
    """Create model with completely deterministic initialization."""
    # Set all random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create model
    model = TorchModelOptimized(input_shape=input_shape, num_classes=num_classes, seed=seed)
    
    # Force evaluation mode to ensure BatchNorm is in consistent state
    model.eval()
    
    # Initialize BatchNorm statistics properly
    # This ensures running_mean and running_var are properly initialized
    with torch.no_grad():
        # Create a dummy input to initialize BatchNorm layers
        dummy_input = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
        _ = model(dummy_input)
    
    return model

def get_model_info(model):
    """Get detailed model information for debugging."""
    info = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layer_info': []
    }
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            info['layer_info'].append({
                'name': name,
                'type': type(module).__name__,
                'params': params
            })
    
    return info