import os
import random
from shutil import copy2

def split_dataset(data_dir, seed=10):
    random.seed(seed)

    train_ratio = 0.64  # 64% for training
    val_ratio = 0.16  # 16% for validation
    test_ratio = 0.20  # 20% for testing

    # Paths for split datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Ensure directories do not already exist
    if os.path.exists(train_dir) or os.path.exists(val_dir) or os.path.exists(test_dir):
        print("WARNING: Training, validation, and/or testing directories already exist. Delete them to rerun the split.")
        return

    # Create train, val, test directories
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    # Iterate through each emotion category (folder)
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir) and class_name not in ['train', 'val', 'test']:  # Ignore existing split folders
            images = [f for f in os.listdir(class_dir) if f.endswith('.png')]  # Get only .png files
            random.shuffle(images)  # Shuffle for randomness

            # Calculate split sizes
            train_size = int(len(images) * train_ratio)
            val_size = int(len(images) * val_ratio)

            # Assign images to train, val, and test sets
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]  # Remaining for testing

            # Create subdirectories for each class
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir)
            os.makedirs(val_class_dir)
            os.makedirs(test_class_dir)

            # Copy images to respective directories
            for img in train_images:
                copy2(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
            for img in val_images:
                copy2(os.path.join(class_dir, img), os.path.join(val_class_dir, img))
            for img in test_images:
                copy2(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

            print(f"Processed {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("\033[1;32mData split completed successfully.\033[0m")

# Run the function
split_dataset("Full Emotions Dataset Spectograms")
