import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image properties
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Updated to 256x256
BATCH_SIZE = 32
NUM_CLASSES = 7  # Surprised, Sad, Neutral, Happy, Fearful, Disgusted, Angry

# Data transforms with enhanced augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

# Paths to dataset
train_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset_224/train"
val_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset_224/val"
test_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset_224/test"

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pretrained ResNet-18 model
resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze early layers, fine-tune last block
for param in resnet_model.parameters():
    param.requires_grad = False  # Freeze all layers

for param in resnet_model.layer4.parameters():  # Unfreeze last ResNet block
    param.requires_grad = True

# Modify the fully connected layer
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # More dropout for regularization
    nn.Linear(512, NUM_CLASSES)
)

# Move model to device
resnet_model = resnet_model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer - Switched to SGD with momentum
optimizer = optim.SGD(resnet_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Train model function
def train_resnet_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=5):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")
    patience_counter = 0  

    for epoch in range(epochs):
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

        # Learning rate update
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        # Save metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        torch.save(metrics, "metrics_resnet.pth")

        # Early stopping logic
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0  
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print(f"Best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies

# Train the model
train_resnet_model(
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=20
)

# Evaluate model function
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
    
    # Save test accuracy to a file
    torch.save({"test_accuracy": test_accuracy}, "test_accuracy.pth")
    
    return test_accuracy

# Evaluate the model
test_accuracy = evaluate_model(resnet_model, test_loader)