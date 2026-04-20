import torch
import torch.nn as nn

# Residual block for 1D CNN
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.Relu1 = nn.ReLU()
        # Second convolutional layer
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)

    # Forward pass with skip connection
    def forward(self, x):
        residual = self.conv1(x)
        out = self.Relu1(residual)
        out = self.conv2(out)
        # Add original input to the output (Residual connection)
        return x + out

# Main Network Architecture
class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        # Initial feature extraction layer
        self.conv1M = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.reluM = nn.ReLU()
        
        # Cascade of 5 residual blocks
        self.residual_block1 = ResidualBlock()
        self.residual_block2 = ResidualBlock()
        self.residual_block3 = ResidualBlock()
        self.residual_block4 = ResidualBlock()
        self.residual_block5 = ResidualBlock()
        
        # Final reconstruction layer
        self.conv2M = nn.Conv1d(64, 1, kernel_size=7, padding=3)

    # Forward pass through the entire network
    def forward(self, x):
        initial_input = x
        
        # Extract features
        out = self.conv1M(x)
        out = self.reluM(out)
        
        # Pass through residual blocks
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        
        # Reconstruct high-resolution audio
        out = self.conv2M(out)
        
        # Global skip connection
        return initial_input + out