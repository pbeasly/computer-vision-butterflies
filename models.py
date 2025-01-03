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
        Args: Image dimensions (int): image_height, image_width
        
        Returns: Classify the image into 10 classes
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

    #----- Begin Block Class
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
        Args:  x: tensor (b, 3, h, w) image
        Returns:  tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        # z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: use the normalized input in your network 'z'
        logits = self.network(x).mean(dim=-1).mean(dim=-1)

        ## logits = torch.randn(x.size(0), 6)

        return logits
    


class UNET(torch.nn.Module):
    #----- Down_Block class here
    class Down_Block(torch.nn.Module):
        # kernel_size = 3, stride = 2, padding = 1
        # Reduce the spatial dimensions by half each time
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, 2, 1)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            # self.norm = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()

            self.network = torch.nn.Sequential(self.conv1, self.relu, self.conv2, self.relu)

        def forward(self, x):
            # TODO: add the norm layer later
            x = self.network(x)
            return x
        
    #----- add the Up_Block class here
    class Up_Block(torch.nn.Module):
        # kernel_size = 3, stride = 2, padding = 1
        # Double the spatial dimensions each time
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1)
            self.norm = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv(x))
            return x
        

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A model that performs segmentation training

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        """Input   (b,  3,     h,     w)    input image
        Down1   (b, 16, h / 2, w / 2)    after strided conv layer
        Down2   (b, 32, h / 4, h / 4)    after strided conv layer
        Up1     (b, 16, h / 2, w / 2)    after up-conv layer
        Up2     (b, 16,     h,     w)    after up-conv layer
        Logits  (b,  n,     h,     w)    output logits, where n = num_class
        Depth   (b,  1,     h,     w)    output depth, single channel
        """
        # --------------- Create the layers

        self.down1 = self.Down_Block(in_channels, 16)
        self.down2 = self.Down_Block(16, 32)
        self.down3 = self.Down_Block(32, 64)
        self.down4 = self.Down_Block(64, 128)

        self.up1 = self.Up_Block(128, 64)
        self.up2 = self.Up_Block(64, 32)
        self.up3 = self.Up_Block(32, 16)
        self.up4 = self.Up_Block(16, 16)
        
        #segmenation and depth heads 
        self.logits = torch.nn.Conv2d(16, num_classes, kernel_size=1)

        self.network = torch.nn.Sequential(self.down1, self.down2, self.up1, self.up2)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # ------------------------------
        """
         # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # TODO: replace with actual forward pass
        logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))
        """

        x1 = self.down1(x)  # (B, 16, H/2, W/2)
        x2 = self.down2(x1)  # (B, 32, H/4, W/4)
        x3 = self.down3(x2)  # (B, 64, H/8, W/8)
        x4 = self.down4(x3)  # (B, 64, H/16, W/16)

        # Decoder
  
        x5 = self.up1(x4)  # (B, 16, H/8, W/8)
        x6 = self.up2(x5)  # (B, 16, H, W)
        x7 = self.up3(x6)  # (B, 16, H, W)
        x8 = self.up4(x7)  # (B, 16, H, W)

        # Output heads
        logits = self.logits(x8)  # (B, num_classes, H, W)



        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth
        # depth = depth.squeeze(dim=1)

        return pred, depth
    

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
    
