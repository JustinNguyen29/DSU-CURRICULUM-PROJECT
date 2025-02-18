import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



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

# Constants
TARGET_SIZE = (224, 224)  # Fixed size for CNN input
NORMALIZE = True  # Normalize pixel values to [0,1]

def save_spectrogram(audio_path, output_path):
    """Generate and save a spectrogram image."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz

        # Create mel-spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

        # Normalize spectrogram to 0-1
        if NORMALIZE:
            mel_spect_db = (mel_spect_db - mel_spect_db.min()) / (mel_spect_db.max() - mel_spect_db.min())

        # Plot the spectrogram
        plt.figure(figsize=(4, 4), dpi=100)
        plt.axis("off")
        librosa.display.specshow(mel_spect_db, sr=sr, hop_length=512, cmap="viridis")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        # Resize the image to the target size
        img = Image.open(output_path)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing
        img.save(output_path)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Process audio files
base_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Emotions Dataset"
output_dir = "/Users/justinnguyen/Desktop/DSU curriculum/ Full Emotions Dataset Spectrograms 224x224"
os.makedirs(output_dir, exist_ok=True)

folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            input_path = os.path.join(folder_path, file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
            save_spectrogram(input_path, output_path)
    print(f"Spectrograms saved for folder: {folder}")
