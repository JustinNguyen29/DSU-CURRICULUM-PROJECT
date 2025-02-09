import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



##### Count the number of audio files in each emotion folder ##### 
base_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Emotions Dataset"

# Ensure the base directory exists
if not os.path.exists(base_dir):
    print(f"Base directory '{base_dir}' does not exist!")
else:
    # List all folders in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print("Emotion Folders:", folders)

    # Count the number of files in each folder
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        # Filter to count only `.wav` files
        wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
        print(f"{folder}: {len(wav_files)} audio files")
####################################################################


# Directories
base_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Emotions Dataset Very Small"
output_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Updated Emotions Dataset Spectograms"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all folders in the base directory
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
print("Processing emotion folders:", folders)

# Constants
TARGET_SIZE = (128, 128)  # Fixed size for CNN input
NORMALIZE = True  # Normalize pixel values to [0,1]
AUGMENTATION = False  # Enable time/frequency masking for augmentation

# Function to apply data augmentation (time/frequency masking)
def augment_spectrogram(mel_spect):
    num_mels, num_frames = mel_spect.shape

    # Time masking
    for _ in range(2):  # Apply twice
        t = np.random.randint(0, num_frames // 5)
        t0 = np.random.randint(0, num_frames - t)
        mel_spect[:, t0:t0 + t] = 0  # Mask time frames

    # Frequency masking
    for _ in range(2):
        f = np.random.randint(0, num_mels // 10)
        f0 = np.random.randint(0, num_mels - f)
        mel_spect[f0:f0 + f, :] = 0  # Mask frequency bands

    return mel_spect

# Function to save a spectrogram while keeping its color
def save_spectrogram(audio_path, output_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create mel-spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Apply augmentation if enabled
        if AUGMENTATION:
            mel_spect_db = augment_spectrogram(mel_spect_db)

        # Normalize to 0-1 range
        if NORMALIZE:
            mel_spect_db = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())

        # Plot spectrogram with color
        plt.figure(figsize=(4, 4))  # Keep square aspect ratio
        plt.axis("off")  # Remove axis for clean image
        librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512, cmap="viridis")

        # Save image
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Process each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)  # Create subfolder for each emotion

    # Process each .wav file
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            input_path = os.path.join(folder_path, file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
            save_spectrogram(input_path, output_path)

    print(f"Spectrograms saved for folder: {folder}")
