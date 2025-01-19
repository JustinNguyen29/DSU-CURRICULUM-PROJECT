import os


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

import librosa 
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Emotions Dataset Very Small"
output_dir = "/Users/justinnguyen/Desktop/DSU curriculum/Emotions Dataset Very Small Spectograms"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all folders in the base directory
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
print("Processing emotion folders:", folders)

# Function to save spectrogram
def save_spectrogram(audio_path, output_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        # Create a mel-spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Plot and save the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
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