import torch
import torch.nn as nn

# Residual block for 1D CNN
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1=nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.Relu1=nn.ReLU()
        self.conv2=nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)

    # Forward pass
    def forward(self, x):
        residual=self.conv1(x)
        out=self.Relu1(residual)
        out=self.conv2(residual)
        return x+ residual

# Main Network
class MainNetwork(nn.Module):
    def __init__(self):
        super(MainNetwork, self).__init__()
        self.conv1M=nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.reluM=nn.ReLU()  
        self.residual_block1=ResidualBlock()
        self.residual_block2=ResidualBlock()
        self.residual_block3=ResidualBlock()
        self.residual_block4=ResidualBlock()
        self.residual_block5=ResidualBlock()
        self.conv2M=nn.Conv1d(64, 1, kernel_size=7, padding=3)

    # Forward pass through network
    def forward(self, x):
        initial_input=x
        out=self.conv1M(x)
        out=self.reluM(out)
        out=self.residual_block1(out)
        out=self.residual_block2(out)
        out=self.residual_block3(out)
        out=self.residual_block4(out)
        out=self.residual_block5(out)
        out=self.conv2M(out)
        return initial_input+out