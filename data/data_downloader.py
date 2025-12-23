import kagglehub
import shutil
import os

# Dictionary mapping kaggle handle to local subfolder
datasets = {
    "fedesoriano/body-fat-prediction-dataset": "body_fat",
    "mathchi/diabetes-data-set": "diabetes",
    # "fabianwinkel/phoenix-contact-relay-dataset": "phoenix_relay"
}

base_dir = "data"

for handle, subfolder in datasets.items():
    print(f"\n--- Processing: {handle} ---")
    
    # 1. Download to kaggle cache
    cache_path = kagglehub.dataset_download(handle)
    
    # 2. destination path
    destination = os.path.join(base_dir, subfolder)
    
    # 3. copy files to local folder
    print(f"Copying files to: {destination}")
    shutil.copytree(cache_path, destination, dirs_exist_ok=True)

print("All datasets downloaded and moved successfully.")
