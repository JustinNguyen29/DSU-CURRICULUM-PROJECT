import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

import ssl
import certifi

# Image properties
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
NUM_CLASSES = 7  # Surprised, Sad, Neutral, Happy, Fearful, Disgusted, Angry
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms with optional data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Paths to your data directories
train_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset/train"
val_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset/val"
test_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset/test"

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Load the pretrained ResNet-18 model
resnet_model = models.resnet18(pretrained=True)

# Replace the fully connected (fc) layer to match your number of classes
num_features = resnet_model.fc.in_features  # Get the number of input features to the fc layer
resnet_model.fc = nn.Linear(num_features, NUM_CLASSES)  # Replace the fc layer

# Move the model to the appropriate device
resnet_model = resnet_model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with weight decay
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-3)


# TRAIN MODEL
def train_resnet_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=3, save_path="metrics_resnet.pth"
):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Initialize the best validation loss to a very large number
    best_val_loss = float("inf")
    patience_counter = 0  # Counter to track patience

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / len(train_loader.dataset))

        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / len(val_loader.dataset))

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        # Early Stopping Logic
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]  # Update the best validation loss
            patience_counter = 0  # Reset patience counter
            # Save the best model state
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print(f"Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}")

        # If patience is exceeded, stop training
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save metrics to a file
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
        },
        save_path,
    )
    print(f"Metrics saved to {save_path}")

    return train_losses, val_losses, train_accuracies, val_accuracies


# Call training function
train_resnet_model(
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=10,
    save_path="metrics_resnet.pth"  # Save to metrics.pth
)


# evalulate model
def evaluate_model(model, test_loader):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

# Evaluate the model
test_accuracy = evaluate_model(resnet_model, test_loader)
