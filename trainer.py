# Different trainers for different models
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from data_loader import ButterflyDataset
from models import MLP, CNNClassifier
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

image_width = 128
image_height = 128

def mlp_train(
    batch_size=64, # seems a bit low as a default, maybe make larger later
    num_epochs=8, # will adjust later
    learning_rate=0.001, 
    pipeline="default",
    weight_decay=1e-4,
    ):
    print("Training MLP model")
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = ButterflyDataset.generate_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    #------ Building graph picture of model
    writer = SummaryWriter()
    model = MLP().to(device)  # Move the model to the same device as the tensor
    writer.add_graph(model, torch.zeros((1, image_width * image_height * 3), device=device))
    writer.flush()
    #--------------------------------------


    mlp_model = MLP().to(device)
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        mlp_model.train()
        train_loss = 0.0
        for i, (images, segmentations, labels) in enumerate(train_loader):
            images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = mlp_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        mlp_model.eval()
        val_loss = 0.0
        correct_predictions = 0  # Initialize to 0
        total_samples = 0  # Initialize to 0
        for i, (images, segmentations, labels) in enumerate(val_loader):
            images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device)
            outputs = mlp_model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples * 100  # Accuracy as a percentage

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)

        writer.flush()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


# CNN model training for classification
def cnn_trainer(
    batch_size=128, 
    num_epochs=10, 
    learning_rate=0.001, 
    weight_decay=1e-4,
    pipeline="default",
):

    print("Training CNN model")
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = ButterflyDataset.generate_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #------ Building graph picture of model
    writer = SummaryWriter()
    # model = CNNClassifier().to(device)  # Move the model to the same device as the tensor
    # writer.add_graph(model, torch.zeros((1, image_width * image_height * 3), device=device))
    # writer.flush()
    #--------------------------------------

    cnn_model = CNNClassifier().to(device)
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        cnn_model.train()
        train_loss = 0.0
        for i, (images, segmentations, labels) in enumerate(train_loader):
            images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        cnn_model.eval()
        val_loss = 0.0
        correct_predictions = 0  # Initialize to 0
        total_samples = 0  # Initialize to 0
        for i, (images, segmentations, labels) in enumerate(val_loader):
            images, segmentations, labels = images.to(device), segmentations.to(device), labels.to(device)
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples * 100  # Accuracy as a percentage

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)

        writer.flush()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the butterfly dataset")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], required=True, help="The model to train: 'mlp' or 'cnn'")
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size for training")
    parser.add_argument("--num_epochs", type=int, default=8, help="The number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate for the optimizer")
    parser.add_argument("--pipeline", type=str, default="default", help="The pipeline to use for training")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="The weight decay for the optimizer")

    args = parser.parse_args()

    if args.model == "mlp":
        mlp_train(args.batch_size, args.num_epochs, args.learning_rate, args.pipeline, args.weight_decay)
    elif args.model == "cnn":
        cnn_trainer(args.batch_size, args.num_epochs, args.learning_rate, args.weight_decay, args.pipeline)
