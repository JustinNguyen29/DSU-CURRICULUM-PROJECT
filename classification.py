import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir = "/Users/justinnguyen/Desktop/DSU curriculum/FULL Emotions Dataset Spectograms"
output_dir = "/Users/justinnguyen/Desktop/DSU curriculum/split_dataset"

# Create train, val, test directories
for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):  # Ensure it's a directory
            os.makedirs(os.path.join(split_path, category), exist_ok=True)

# Split data
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    
    # Ensure it's a directory
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        
        # Filter out non-image files (e.g., hidden files)
        images = [img for img in images if img.endswith('.png')]  # Adjust the extension if needed
        
        # Split the data
        train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "train", category, img))
        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "val", category, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_dir, "test", category, img))

        print(f"Processed {category}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("Dataset split completed successfully.")