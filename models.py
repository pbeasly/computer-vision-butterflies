# Models for training and testing the model
# minor change for git control testing
from pathlib import Path

import torch
import torch.nn as nn

image_width = 128
image_height = 128

class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x) + x)
        
    def __init__(
        self,
        hidden_dim: int = 256,
    ):
        """
        Args:
            image dimensions (int): image_height, image_width
        
        Returns:
            Classify the image into 10 classes
        """
        super().__init__()
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(3 * image_height * image_width, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.25))  # Dropout after the first activation
        layers.append(self.Block(hidden_dim, hidden_dim))
        layers.append(self.Block(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 10))

        self.mlp_model = nn.Sequential(*layers)

    def forward(self, x):
        """ Predicts the classification of the image"""
        return self.mlp_model(x)


""" This is a simpler model that can be used as a reference"""
# Create a simple perceptron model
class MLPSimple(torch.nn.Module):
    def __init__(self, layer_size = [512, 512, 512]):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = image_width*image_height*3

        # operation is 128*128*3 -> 512 ->512 -> 512 -> 10
        for s in layer_size:
            layers.append(torch.nn.Linear(c, s))
            layers.append(torch.nn.ReLU())
            c = s

        # output layer
        layers.append(torch.nn.Linear(c, 10))

        # create netwrork that feeds the output of one layer to the next
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
