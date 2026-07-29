import os
import json
import yaml
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, images_dir, labels_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Create directories
    os.makedirs(labels_dir, exist_ok=True)

    # Category mapping to 0-indexed YOLO format
    categories = data.get('categories', [])
    categories.sort(key=lambda x: x['id'])
    category_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
    yolo_id_to_name = {i: cat['name'] for i, cat in enumerate(categories)}

    # Image mapping
    images = data.get('images', [])
    image_id_to_info = {img['id']: img for img in images}

    # Group annotations by image
    annotations = data.get('annotations', [])
    img_to_anns = {img['id']: [] for img in images}
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    for img_id, anns in img_to_anns.items():
        img_info = image_id_to_info[img_id]
        img_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        with open(label_path, 'w') as f_out:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in category_id_to_yolo_id:
                    continue
                yolo_class_id = category_id_to_yolo_id[cat_id]
                
                # YOLO segmentation format: class_id x1 y1 x2 y2 ... (normalized)
                if 'segmentation' in ann and isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    for poly in ann['segmentation']:
                        # Skip empty polygons
                        if len(poly) == 0:
                            continue
                        
                        normalized_poly = []
                        # coordinates are [x1, y1, x2, y2, ...]
                        for i in range(0, len(poly), 2):
                            x = poly[i] / img_w
                            y = poly[i+1] / img_h
                            normalized_poly.extend([f"{x:.6f}", f"{y:.6f}"])
                        
                        f_out.write(f"{yolo_class_id} " + " ".join(normalized_poly) + "\n")
                
                # Fallback to bbox if segmentation is empty or not present
                elif 'bbox' in ann:
                    bbox = ann['bbox'] # [x_min, y_min, width, height]
                    x_center = (bbox[0] + bbox[2] / 2) / img_w
                    y_center = (bbox[1] + bbox[3] / 2) / img_h
                    norm_w = bbox[2] / img_w
                    norm_h = bbox[3] / img_h
                    f_out.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    return yolo_id_to_name

def main():
    base_dir = Path("data/car_DD_damage")
    yolo_dir = Path("data/car_DD_damage_yolo")
    
    splits = ['train', 'val', 'test']
    all_classes = {}
    
    for split in splits:
        json_path = base_dir / f"{split}.json"
        if not json_path.exists():
            print(f"Skipping {split}, {json_path} not found.")
            continue
            
        print(f"Converting {split} split...")
        images_dir = base_dir / split
        labels_dir = yolo_dir / "labels" / split
        dest_images_dir = yolo_dir / "images" / split
        
        classes = convert_coco_to_yolo(json_path, dest_images_dir, labels_dir)
        all_classes.update(classes)
        
        # Link or copy images to the new YOLO directory structure
        os.makedirs(dest_images_dir, exist_ok=True)
        # Using a symbolic link or copying is better, but since it's windows, we can just use relative paths in data.yaml
        # Or copy them using shutil to keep standard YOLO structure
        import shutil
        for img in os.listdir(images_dir):
            if img.endswith('.jpg') or img.endswith('.png'):
                src = images_dir / img
                dst = dest_images_dir / img
                if not dst.exists():
                    shutil.copy2(src, dst)

    # Generate data.yaml
    if all_classes:
        class_names = [all_classes[i] for i in range(len(all_classes))]
        yaml_content = {
            'path': str(yolo_dir.absolute()), # Absolute path to be safe
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': class_names
        }
        
        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"\nConversion complete! Created YOLO dataset at {yolo_dir.absolute()}")
        print(f"YOLO data.yaml saved to: {yaml_path.absolute()}")
        print(f"Classes: {class_names}")

if __name__ == "__main__":
    main()
