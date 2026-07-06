import json
import os
import shutil
from pathlib import Path
import random
import yaml

def convert_supervisely_to_yolo(sly_dataset_dir, output_dir, split_ratio=0.8):
    """
    Converts a Supervisely instance segmentation dataset to YOLO segmentation format.
    """
    sly_dataset_dir = Path(sly_dataset_dir)
    output_dir = Path(output_dir)
    
    # 1. Read meta.json
    meta_path = sly_dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {sly_dataset_dir}")
        
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    class_titles = [cls['title'] for cls in meta.get('classes', [])]
    class_title_to_idx = {title: idx for idx, title in enumerate(class_titles)}
    
    print(f"Found {len(class_titles)} classes in meta.json")
    
    # 2. Create YOLO directory structure
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    dataset_paths = [d for d in sly_dataset_dir.iterdir() if d.is_dir() and (d / 'ann').exists()]
    
    # 3. Collect all image and annotation pairs
    all_items = []
    for ds_path in dataset_paths:
        ann_dir = ds_path / 'ann'
        img_dir = ds_path / 'img'
        
        for ann_file in ann_dir.glob('*.json'):
            # The image file is typically named by removing the trailing `.json` from the annotation file name
            img_file_name = ann_file.name.replace('.json', '')
            img_file = img_dir / img_file_name
            if img_file.exists():
                all_items.append((img_file, ann_file))
                
    if not all_items:
        print("No image/annotation pairs found!")
        return
        
    # 4. Shuffle and split into train/val
    random.seed(42)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * split_ratio)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    def process_items(items, split):
        for img_path, ann_path in items:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
                
            img_h = ann_data['size']['height']
            img_w = ann_data['size']['width']
            
            # Destination paths
            dst_img = output_dir / 'images' / split / img_path.name
            dst_lbl = output_dir / 'labels' / split / f"{img_path.stem}.txt"
            
            # Copy image
            shutil.copy(img_path, dst_img)
            
            # Create label text file
            with open(dst_lbl, 'w', encoding='utf-8') as f_lbl:
                for obj in ann_data.get('objects', []):
                    cls_title = obj.get('classTitle')
                    if cls_title not in class_title_to_idx:
                        continue
                        
                    cls_idx = class_title_to_idx[cls_title]
                    points = obj.get('points', {}).get('exterior', [])
                    if not points:
                        continue
                        
                    # Normalize points (x / width, y / height)
                    normalized_points = []
                    for x, y in points:
                        nx = max(0.0, min(1.0, x / img_w))
                        ny = max(0.0, min(1.0, y / img_h))
                        normalized_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
                        
                    # Write in YOLO format: <class_id> <x1> <y1> <x2> <y2> ...
                    if len(normalized_points) >= 6:  # Need at least 3 points for a valid polygon
                        f_lbl.write(f"{cls_idx} {' '.join(normalized_points)}\n")
                        
    print(f"Processing {len(train_items)} training images...")
    process_items(train_items, 'train')
    print(f"Processing {len(val_items)} validation images...")
    process_items(val_items, 'val')
    
    # 5. Create dataset.yaml configuration file
    yaml_data = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {idx: name for name, idx in class_title_to_idx.items()}
    }
    
    with open(output_dir / 'dataset.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"\nConversion complete! YOLO dataset saved to {output_dir}")
    print(f"You can now train using this command:")
    print(f"yolo segment train data=\"{output_dir / 'dataset.yaml'}\" model=yolo11n-seg.pt epochs=50 imgsz=640")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert Supervisely to YOLO format")
    parser.add_argument('--input', type=str, required=True, help="Path to the Supervisely dataset root (containing meta.json)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the YOLO dataset")
    parser.add_argument('--split', type=float, default=0.8, help="Train split ratio (default 0.8)")
    
    args = parser.parse_args()
    convert_supervisely_to_yolo(args.input, args.output, args.split)
