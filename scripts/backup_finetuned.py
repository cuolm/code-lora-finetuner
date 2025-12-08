import os
import shutil
from datetime import datetime
from pathlib import Path

def backup_folders(folders, project_root_path):
    backup_path = f"{project_root_path}/backup"
    os.makedirs(backup_path, exist_ok=True)
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = os.path.join(backup_path, f"backup_{now_str}")
    os.makedirs(backup_folder, exist_ok=True)
    
    for folder in folders:
        if os.path.exists(folder):
            base_name = os.path.basename(os.path.normpath(folder))
            renamed_folder = f"{base_name}_{now_str}"
            dest_folder = os.path.join(backup_folder, renamed_folder)
            shutil.copytree(folder, dest_folder)
            print(f"Copied and renamed {folder} to {dest_folder}")
        else:
            print(f"Folder {folder} does not exist. Skipping.")

if __name__ == "__main__":
    project_root_path = Path(__file__).resolve().parent.parent
    folders_to_backup = [
        f"{project_root_path}/src", 
        f"{project_root_path}/results",
        f"{project_root_path}/lora_model", 
        f"{project_root_path}/lora_adapter",
        f"{project_root_path}/datasets"
        ]
    backup_folders(folders_to_backup, project_root_path)
