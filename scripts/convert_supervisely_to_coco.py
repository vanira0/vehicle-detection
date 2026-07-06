import json
import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

def create_coco_structure(categories):
    return {
        "info": {
            "description": "Converted from Supervisely",
            "version": "1.0",
            "year": 2026
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories
    }

def convert_supervisely_to_coco(sly_dataset_dir, output_dir, split_ratio=0.8):
    sly_dataset_dir = Path(sly_dataset_dir)
    output_dir = Path(output_dir)
    
    # Read meta.json
    meta_path = sly_dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {sly_dataset_dir}")
        
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    class_titles = [cls['title'] for cls in meta.get('classes', [])]
    
    # COCO categories start at 1
    categories = []
    class_title_to_id = {}
    for idx, title in enumerate(class_titles, 1):
        categories.append({"id": idx, "name": title, "supercategory": "none"})
        class_title_to_id[title] = idx
        
    print(f"Found {len(categories)} categories in meta.json")
    
    # Create COCO directory structure
    for split in ['train', 'valid']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        
    dataset_paths = [d for d in sly_dataset_dir.iterdir() if d.is_dir() and (d / 'ann').exists()]
    
    all_items = []
    for ds_path in dataset_paths:
        ann_dir = ds_path / 'ann'
        img_dir = ds_path / 'img'
        
        for ann_file in ann_dir.glob('*.json'):
            img_file_name = ann_file.name.replace('.json', '')
            img_file = img_dir / img_file_name
            if img_file.exists():
                all_items.append((img_file, ann_file))
                
    if not all_items:
        print("No image/annotation pairs found!")
        return
        
    random.seed(42)
    random.shuffle(all_items)
    split_idx = int(len(all_items) * split_ratio)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    def process_items(items, split):
        coco_data = create_coco_structure(categories)
        annotation_id = 1
        
        for image_id, (img_path, ann_path) in enumerate(items, 1):
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
                
            img_h = ann_data['size']['height']
            img_w = ann_data['size']['width']
            
            # Copy image
            dst_img = output_dir / split / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Add to COCO images
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h
            })
            
            # Add to COCO annotations
            for obj in ann_data.get('objects', []):
                cls_title = obj.get('classTitle')
                if cls_title not in class_title_to_id:
                    continue
                    
                category_id = class_title_to_id[cls_title]
                points = obj.get('points', {}).get('exterior', [])
                if not points or len(points) < 3:
                    continue
                
                # Format points for COCO: [x1, y1, x2, y2, ...]
                segmentation = []
                contour = []
                for x, y in points:
                    # Clip to image boundaries
                    px = max(0, min(img_w - 1, int(x)))
                    py = max(0, min(img_h - 1, int(y)))
                    segmentation.extend([px, py])
                    contour.append([px, py])
                
                if len(segmentation) < 6:
                    continue
                    
                contour_np = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
                area = float(cv2.contourArea(contour_np))
                x, y, w, h = cv2.boundingRect(contour_np)
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0
                })
                annotation_id += 1
                
        # Write COCO JSON
        json_path = output_dir / split / "_annotations.coco.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, separators=(',', ':'))
            
        return len(items)

    print(f"Processing {len(train_items)} training images...")
    processed_train = process_items(train_items, 'train')
    
    print(f"Processing {len(val_items)} validation images...")
    processed_val = process_items(val_items, 'valid')
    
    print(f"\nConversion complete! COCO dataset saved to {output_dir}")
    print(f"Train images: {processed_train}, Validation images: {processed_val}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert Supervisely to COCO format")
    parser.add_argument('--input', type=str, required=True, help="Path to the Supervisely dataset root (containing meta.json)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the COCO dataset")
    parser.add_argument('--split', type=float, default=0.8, help="Train split ratio (default 0.8)")
    
    args = parser.parse_args()
    convert_supervisely_to_coco(args.input, args.output, args.split)
