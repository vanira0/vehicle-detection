import os
import argparse
import subprocess
import zipfile
import json
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Instructions for setting up Kaggle API credentials."""
    kaggle_dir = Path(".kaggle")
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("Error: kaggle.json not found!")
        print("To download datasets from Kaggle, you need an API token.")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll down to 'API' section and click 'Create New Token'")
        print("3. A 'kaggle.json' file will be downloaded.")
        print(f"4. Move this file to: {kaggle_json}")
        print("5. Run this script again.")
        return False
    return True

def convert_coco_to_yolo(coco_json_path, output_dir):
    """
    Converts COCO format to YOLO format.
    Uses ultralytics built-in converter if available.
    """
    try:
        from ultralytics.data.converter import convert_coco
        print(f"Converting COCO annotations from {coco_json_path} to YOLO format...")
        convert_coco(labels_dir=str(Path(coco_json_path).parent), save_dir=output_dir, use_segments=True)
    except ImportError:
        print("Ultralytics package is required for COCO to YOLO conversion.")
        print("Please install it with: pip install ultralytics")
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Note: The Kaggle dataset might not be in standard COCO format.")
        print("You may need to write a custom JSON parser based on its specific structure.")

def download_dataset(dataset_name, dest_dir="data"):
    """Downloads and unzips a Kaggle dataset."""
    dest_path = Path(dest_dir) / dataset_name.split('/')[-1]
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name} to {dest_path}...")
    
    try:
        # Run kaggle CLI command
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(Path(".kaggle").absolute())
        
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name, "-p", str(dest_path)
        ], check=True, env=env)
        
        # Unzip the downloaded file
        zip_file = dest_path / f"{dataset_name.split('/')[-1]}.zip"
        if zip_file.exists():
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
            
            # Remove zip file after extraction
            zip_file.unlink()
            print("Extraction complete.")
            
            # Check for JSON files that might need conversion
            json_files = list(dest_path.rglob("*.json"))
            if json_files:
                print(f"Found {len(json_files)} JSON annotation files.")
                print("If these are in COCO format, you can convert them to YOLO format for YOLO11 training.")
                # We don't automatically convert because Kaggle formats vary wildly.
                # However, providing a pointer to the conversion function helps.
                
    except subprocess.CalledProcessError:
        print(f"Failed to download dataset. Please check if the dataset name '{dataset_name}' is correct and you have accepted its rules on Kaggle.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle")
    parser.add_argument("--dataset", type=str, default="humansintheloop/car-parts-and-car-damages",
                        help="Kaggle dataset identifier (e.g., humansintheloop/car-parts-and-car-damages)")
    parser.add_argument("--dest", type=str, default="data/kaggle",
                        help="Destination directory")
    
    args = parser.parse_args()
    
    if setup_kaggle_credentials():
        download_dataset(args.dataset, args.dest)
