import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image properties and batch size
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = 7  # Surprised, Sad, Neutral, Happy, Fearful, Disgusted, Angry

# Data transforms with enhanced augmentation for training and ImageNet normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_HEIGHT, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

# Dataset paths (update if needed)
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
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Unfreeze all layers for full fine-tuning
for param in resnet_model.parameters():
    param.requires_grad = True

# Modify the fully connected layer: reduced dropout from 0.6 to 0.4
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES)
)
resnet_model = resnet_model.to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()

# Group parameters for differential learning rates: lower for backbone, higher for new head layers
backbone_params = []
head_params = []
for name, param in resnet_model.named_parameters():
    if "fc" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

# Use AdamW optimizer with parameter groups:
# Backbone with lr=1e-4
# Head with lr=1e-3
optimizer = optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3},
], weight_decay=1e-4)

# Cosine annealing learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# Train model function with early stopping
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

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # Save metrics
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        torch.save(metrics, "metrics_resnet.pth")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
    torch.save({"test_accuracy": test_accuracy}, "test_accuracy.pth")
    return test_accuracy

# Train the model
train_resnet_model(
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=20,
    patience=5
)

# Evaluate the model
test_accuracy = evaluate_model(resnet_model, test_loader)
