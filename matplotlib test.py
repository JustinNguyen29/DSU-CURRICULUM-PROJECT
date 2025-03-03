import torch
import matplotlib.pyplot as plt

# Function to load metrics and plot
def load_and_plot_metrics(metrics_path):
    # Load the correct metrics file
    metrics = torch.load(metrics_path)

    # Extract the metrics
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    train_accuracies = metrics["train_accuracies"]
    val_accuracies = metrics["val_accuracies"]

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

# Specify the **correct** path to the metrics.pth file
metrics_file = "metrics_resnet.pth" 

# Call the function to load and plot metrics
load_and_plot_metrics(metrics_file)
