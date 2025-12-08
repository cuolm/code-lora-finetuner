import os
import zipfile
from pathlib import Path

def unzip_all_archives(directory_path: str):
    base_directory = Path(directory_path).resolve()

    if not base_directory.is_dir():
        print(f"Error: Directory not found at {base_directory}")
        return

    zip_files_to_process = []

    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename.endswith(".zip"):
                zip_files_to_process.append(Path(root) / filename)

    if not zip_files_to_process:
        print("No zip files found.")
        return

    for zip_filepath in zip_files_to_process:
        destination_directory = zip_filepath.parent

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(destination_directory)

        print(f"Extraction of {zip_filepath.name} successful.")
    
    print(f"Finished processing {len(zip_files_to_process)} archives.")

def main():
    project_root_path = Path(__file__).resolve().parent.parent
    raw_data_path = project_root_path / "data"
    unzip_all_archives(raw_data_path)

if __name__ == "__main__":
    main()
