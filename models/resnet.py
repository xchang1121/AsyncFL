# models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def ResNet18_CIFAR(num_classes: int = 10) -> nn.Module:
    """
    Constructs a ResNet-18 model adapted for CIFAR-10 (32x32 images).

    Args:
        num_classes (int): Number of output classes. Default is 10 for CIFAR-10.

    Returns:
        nn.Module: The ResNet-18 model.
    """
    # Load pretrained ResNet-18 or initialize weights randomly
    # Setting pretrained=False as we typically train from scratch in FL sims
    model = resnet18(weights=None, num_classes=num_classes) 

    # Modify the first convolutional layer for 32x32 input images (CIFAR-10)
    # Original ResNet C1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adapted C1: Use smaller kernel and stride
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the initial max pooling layer, as it's too aggressive for 32x32
    # Original ResNet MaxPool: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # We replace it with an identity layer (does nothing)
    model.maxpool = nn.Identity() 

    # The final fully connected layer (fc) is already adjusted by num_classes in resnet18
    # model.fc = nn.Linear(model.fc.in_features, num_classes) # This line is redundant if using num_classes argument above

    return model

# Example Usage
if __name__ == '__main__':
    # Create a model for CIFAR-10 (10 classes)
    model_cifar10 = ResNet18_CIFAR(num_classes=10)
    print("--- ResNet-18 for CIFAR-10 ---")
    # print(model_cifar10) # Optional: Print model structure

    # Test with a dummy input tensor (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model_cifar10(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [4, 10]
    assert output.shape == (4, 10)

    # Create a model for CIFAR-100 (100 classes)
    model_cifar100 = ResNet18_CIFAR(num_classes=100)
    print("\n--- ResNet-18 for CIFAR-100 ---")
    output_100 = model_cifar100(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output_100.shape}") # Should be [4, 100]
    assert output_100.shape == (4, 100)