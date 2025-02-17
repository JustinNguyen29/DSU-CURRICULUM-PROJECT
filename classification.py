import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# # Define paths
# dataset_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Full Emotions Dataset Spectrograms 228x228"
# output_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset_228"

# # Create train, val, test directories
# for split in ["train", "val", "test"]:
#     split_path = os.path.join(output_dir, split)
#     os.makedirs(split_path, exist_ok=True)

#     for category in os.listdir(dataset_dir):
#         category_path = os.path.join(dataset_dir, category)
#         if os.path.isdir(category_path):  # Ensure it's a directory
#             os.makedirs(os.path.join(split_path, category), exist_ok=True)

# # Split data
# train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

# for category in os.listdir(dataset_dir):
#     category_path = os.path.join(dataset_dir, category)
#     if not os.path.isdir(category_path):  # Skip non-directory files
#         continue
    
#     images = os.listdir(category_path)
#     train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
#     val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

#     for img in train_images:
#         shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "train", category, img))
#     for img in val_images:
#         shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "val", category, img))
#     for img in test_images:
#         shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "test", category, img))

# print("Dataset split completed successfully.")


# Paths to the dataset
dataset_path = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset_228"
train_dir = f"{dataset_path}/train"
val_dir = f"{dataset_path}/val"
test_dir = f"{dataset_path}/test"

# Image properties
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
NUM_CLASSES = 7  # Surprised, Sad, Neutral, Happy, Fearful, Disgusted, Angry
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (preprocessing and augmentation)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
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

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Initialize the model, loss function, and optimizer
model = EmotionCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# Training loop
# Training loop with metrics saving
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
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

        val_loss = 0
        val_correct = 0
        model.eval()
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

    # Save metrics to a file
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, 'metrics.pth')

    print("Metrics saved to 'metrics.pth'")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Test the model
def test_model(model, test_loader):
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            test_correct += (outputs.argmax(1) == labels).sum().item()
    
    print(f"Test Accuracy: {test_correct / len(test_loader.dataset):.4f}")

test_model(model, test_loader)

# Save the model
torch.save(model.state_dict(), "emotion_cnn_model_updated.pth")

