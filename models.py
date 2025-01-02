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

class CNNClassifier(nn.Module):

    #----- add the Block class here
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2

            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.n1 = torch.nn.BatchNorm2d(out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.n2 = torch.nn.BatchNorm2d(out_channels)
            self.relu1 = torch.nn.ReLU()
            self.relu2 = torch.nn.ReLU()

            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()

        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            return self.skip(x0) + x
    #----- end of Block class

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        # self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        # self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # This is the initial layer of the network
        cnn_layers = [
            torch.nn.Conv2d(in_channels, 64, kernel_size=11, stride=2, padding=5),
            torch.nn.ReLU(),
        ]

        # This is the block of layers that will be repeated
        # The number of channels will be doubled each time
        c1 = 64
        num_blocks = 2
        for _ in range(num_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.network = torch.nn.Sequential(*cnn_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        # z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: use the normalized input in your network 'z'
        logits = self.network(x).mean(dim=-1).mean(dim=-1)

        ## logits = torch.randn(x.size(0), 6)

        return logits
    


""" This is a simpler model that can be used as a reference for MLP"""
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
    
