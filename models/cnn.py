# models/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_CIFAR(nn.Module):
    """
    A standard Convolutional Neural Network model for CIFAR-10 classification.
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> FC -> ReLU -> FC
    """
    def __init__(self, num_classes: int = 10):
        """
        Initializes the CNN model.

        Args:
            num_classes (int): Number of output classes. Default is 10 for CIFAR-10.
        """
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features after conv and pool layers
        # Input: 3x32x32
        # After conv1 (32x32x32), pool1 (32x16x16)
        # After conv2 (64x16x16), pool2 (64x8x8)
        self._fc1_input_features = 64 * 8 * 8 
        
        self.fc1 = nn.Linear(self._fc1_input_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output tensor (batch_size, num_classes).
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, self._fc1_input_features) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Output layer (logits)
        return x

# Example Usage
if __name__ == '__main__':
    # Create a model for CIFAR-10 (10 classes)
    model_cnn_cifar10 = CNN_CIFAR(num_classes=10)
    print("--- CNN for CIFAR-10 ---")
    # print(model_cnn_cifar10) # Optional: Print model structure

    # Test with a dummy input tensor (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model_cnn_cifar10(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [4, 10]
    assert output.shape == (4, 10)

    # Create a model for a different number of classes (e.g., 100)
    model_cnn_cifar100 = CNN_CIFAR(num_classes=100)
    print("\n--- CNN for CIFAR-100 ---")
    output_100 = model_cnn_cifar100(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output_100.shape}") # Should be [4, 100]
    assert output_100.shape == (4, 100)