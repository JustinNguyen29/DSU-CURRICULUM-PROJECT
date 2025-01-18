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

##TESTING GITHUB REPOSIT