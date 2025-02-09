import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "Emotions Dataset Spectograms"
output_dir = "split_dataset"

# Create train, val, test directories
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

    for category in os.listdir(dataset_dir):
        os.makedirs(os.path.join(split_path, category), exist_ok=True)

# Split data
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "train", category, img))
    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "val", category, img))
    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "test", category, img))

print("Dataset split completed successfully.")
