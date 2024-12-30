import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split

class ButterflyDataset(Dataset):
    def __init__(self, image_dir, segmentation_dir, transform=None, seg_transform=None):
        """
        Args:
            image_dir (str): Path to the image directory.
            segmentation_dir (str): Path to the segmentation directory.
            transform (callable, optional): Transform to be applied to images.
            seg_transform (callable, optional): Transform to be applied to segmentations.
        """
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.transform = transform
        self.seg_transform = seg_transform
        self.image_paths = []
        self.segmentation_paths = []
        self.labels = []

        # Traverse the image directory to collect image and segmentation paths
        for file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, file)
            segmentation_path = os.path.join(segmentation_dir, file.replace(".", "_seg0."))

            if os.path.exists(image_path) and os.path.exists(segmentation_path):
                self.image_paths.append(image_path)
                self.segmentation_paths.append(segmentation_path)
                # Normalize labels: map '001', '002', ..., '010' to integers 0, 1, ..., 9
                label = int(file[:3]) - 1
                self.labels.append(label)
            else:
                print(f"Warning: Missing image or segmentation for {file}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        # Load the segmentation
        seg_path = self.segmentation_paths[idx]
        segmentation = Image.open(seg_path)

        # Get the label
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.seg_transform:
            segmentation = self.seg_transform(segmentation)

        return image, segmentation, label

    @classmethod
    def generate_datasets(cls, image_dir="dataset/leedsbutterfly/images", 
                          segmentation_dir="dataset/leedsbutterfly/segmentations"):
        """
        Generates training and validation datasets.

        Args:
            image_dir (str): Path to the image directory.
            segmentation_dir (str): Path to the segmentation directory.

        Returns:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
        """
        # Define transformations - resize to 128x128 and convert to tensor
        image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        segmentation_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Instantiate the dataset
        dataset = cls(
            image_dir=image_dir,
            segmentation_dir=segmentation_dir,
            transform=image_transform,
            seg_transform=segmentation_transform
        )

        # Calculate lengths for training and validation splits
        split = 0.8
        dataset_length = len(dataset)
        train_length = int(split * dataset_length)
        val_length = dataset_length - train_length

        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

        return train_dataset, val_dataset
