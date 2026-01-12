#!/usr/bin/env python3

import json
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def merge_coco_datasets(labeled_json, pseudo_json, output_json, 
                       labeled_img_dir, pseudo_img_dir, output_img_dir):
    print("Merging datasets")
    
    with open(labeled_json, 'r') as f:
        labeled_data = json.load(f)
    
    with open(pseudo_json, 'r') as f:
        pseudo_data = json.load(f)
    
    print(f"Labeled data: {len(labeled_data['images'])} images, {len(labeled_data['annotations'])} annotations")
    print(f"Pseudo-labeled data: {len(pseudo_data['images'])} images, {len(pseudo_data['annotations'])} annotations")
    
    # Create merged dataset
    merged_data = {
        'info': labeled_data.get('info', {}),
        'licenses': labeled_data.get('licenses', []),
        'categories': labeled_data['categories'],
        'images': [],
        'annotations': []
    }
    
    # Track new IDs
    img_id_offset = 0
    ann_id_offset = 0
    
    # Add labeled data first (keep as is)
    merged_data['images'].extend(labeled_data['images'])
    merged_data['annotations'].extend(labeled_data['annotations'])
    
    img_id_offset = max([img['id'] for img in labeled_data['images']]) if labeled_data['images'] else 0
    ann_id_offset = max([ann['id'] for ann in labeled_data['annotations']]) if labeled_data['annotations'] else 0
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Copy labeled images
    print(f"\nCopying labeled images...")
    for img_info in tqdm(labeled_data['images']):
        src = os.path.join(labeled_img_dir, img_info['file_name'])
        dst = os.path.join(output_img_dir, img_info['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # Add pseudo-labeled data with adjusted IDs
    print(f"\nCopying pseudo-labeled images...")
    for img_info in tqdm(pseudo_data['images']):
        new_img = img_info.copy()
        new_img['id'] = img_info['id'] + img_id_offset
        merged_data['images'].append(new_img)
        
        # Copy image
        src = os.path.join(pseudo_img_dir, img_info['file_name'])
        dst = os.path.join(output_img_dir, img_info['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    print("Merging annotations...")
    for ann in tqdm(pseudo_data['annotations']):
        new_ann = ann.copy()
        new_ann['id'] = ann['id'] + ann_id_offset
        new_ann['image_id'] = ann['image_id'] + img_id_offset
        merged_data['annotations'].append(new_ann)
    
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print("Merge complete")
    print(f"Merged dataset: {len(merged_data['images'])} images, {len(merged_data['annotations'])} annotations")
    print(f"Output: {output_json}")

def main():
    parser = argparse.ArgumentParser(description='Merge labeled and pseudo-labeled datasets')
    parser.add_argument('--labeled_json', type=str, required=True,
                       help='Path to labeled dataset JSON')
    parser.add_argument('--pseudo_json', type=str, required=True,
                       help='Path to pseudo-labeled dataset JSON')
    parser.add_argument('--output_json', type=str, required=True,
                       help='Output path for merged JSON')
    parser.add_argument('--labeled_img_dir', type=str, required=True,
                       help='Directory containing labeled images')
    parser.add_argument('--pseudo_img_dir', type=str, required=True,
                       help='Directory containing pseudo-labeled images')
    parser.add_argument('--output_img_dir', type=str, required=True,
                       help='Output directory for merged images')
    
    args = parser.parse_args()
    
    merge_coco_datasets(
        args.labeled_json,
        args.pseudo_json,
        args.output_json,
        args.labeled_img_dir,
        args.pseudo_img_dir,
        args.output_img_dir
    )

if __name__ == '__main__':
    main()




