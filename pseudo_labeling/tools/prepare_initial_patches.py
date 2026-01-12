#!/usr/bin/env python3

import json
import os
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

def crop_image_to_patches(image_path, output_dir, patch_size=341, overlap=10):
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    def _compute_offsets(length):
        stride = max(patch_size - overlap, 1)
        offsets = list(range(0, max(length - patch_size, 0) + 1, stride))
        if not offsets:
            offsets = [0]
        elif offsets[-1] + patch_size < length:
            offsets.append(length - patch_size)
        return offsets

    x_offsets = _compute_offsets(img_width)
    y_offsets = _compute_offsets(img_height)

    patches = []
    patch_id = 0

    for y_offset in y_offsets:
        for x_offset in x_offsets:
            x_end = min(x_offset + patch_size, img_width)
            y_end = min(y_offset + patch_size, img_height)

            patch = img.crop((x_offset, y_offset, x_end, y_end))

            img_name = Path(image_path).stem
            patch_name = f"{img_name}_patch{patch_id}.jpg"
            patch_path = os.path.join(output_dir, patch_name)
            patch.save(patch_path, quality=95)

            patches.append({
                'patch_name': patch_name,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'width': x_end - x_offset,
                'height': y_end - y_offset,
                'patch_id': patch_id
            })

            patch_id += 1

    return patches

def adjust_annotation(ann, x_offset, y_offset, patch_width, patch_height):
    x, y, w, h = ann['bbox']
    
    # Calculate bbox bounds
    ann_x1, ann_y1 = x, y
    ann_x2, ann_y2 = x + w, y + h
    
    patch_x1, patch_y1 = x_offset, y_offset
    patch_x2, patch_y2 = x_offset + patch_width, y_offset + patch_height
    
    # Calculate intersection
    int_x1 = max(ann_x1, patch_x1)
    int_y1 = max(ann_y1, patch_y1)
    int_x2 = min(ann_x2, patch_x2)
    int_y2 = min(ann_y2, patch_y2)
    
    # Check if there's overlap
    if int_x2 <= int_x1 or int_y2 <= int_y1:
        return None
    
    # Check if >50% of annotation is in the patch
    ann_area = w * h
    int_area = (int_x2 - int_x1) * (int_y2 - int_y1)
    
    if int_area / ann_area < 0.5:
        return None
    
    # Adjust bbox coordinates
    new_x = int_x1 - x_offset
    new_y = int_y1 - y_offset
    new_w = int_x2 - int_x1
    new_h = int_y2 - int_y1
    
    # Create adjusted annotation
    new_ann = ann.copy()
    new_ann['bbox'] = [new_x, new_y, new_w, new_h]
    new_ann['area'] = new_w * new_h
    
    # Adjust segmentation if present
    if 'segmentation' in ann and ann['segmentation']:
        new_segs = []
        for seg in ann['segmentation']:
            new_seg = []
            for i in range(0, len(seg), 2):
                seg_x = seg[i] - x_offset
                seg_y = seg[i + 1] - y_offset
                
                # Clip to patch boundaries
                seg_x = max(0, min(seg_x, patch_width))
                seg_y = max(0, min(seg_y, patch_height))
                
                new_seg.extend([seg_x, seg_y])
            
            if len(new_seg) >= 6:  # Valid polygon needs at least 3 points
                new_segs.append(new_seg)
        
        if new_segs:
            new_ann['segmentation'] = new_segs
        else:
            return None
    
    return new_ann

def create_patched_dataset(data_root, output_root, split='train', patch_size=341, overlap=10):
    print(f"Processing {split.upper()} split")
    print(f"Input:  {data_root}/{split}/")
    print(f"Output: {output_root}/{split}/")
    
    ann_file = os.path.join(data_root, split, f'{split}.json')
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return
        
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_img_dir = os.path.join(output_root, split)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # New COCO data
    new_coco = {
        'info': coco_data.get('info', {'description': 'Patched Stomata Dataset'}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': [],
        'annotations': []
    }
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image
    new_image_id = 1
    new_ann_id = 1
    
    for img_info in tqdm(coco_data['images'], desc=f"Patching {split} images"):
        img_id = img_info['id']
        img_path = os.path.join(data_root, split, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Crop image into patches
        patches = crop_image_to_patches(img_path, output_img_dir, patch_size, overlap)
        
        # Get annotations for this image
        img_annotations = annotations_by_image.get(img_id, [])
        
        # Process each patch
        for patch_info in patches:
            # Adjust annotations for this patch
            patch_annotations = []
            for ann in img_annotations:
                adjusted_ann = adjust_annotation(
                    ann,
                    patch_info['x_offset'],
                    patch_info['y_offset'],
                    patch_info['width'],
                    patch_info['height']
                )
                
                if adjusted_ann:
                    adjusted_ann['id'] = new_ann_id
                    adjusted_ann['image_id'] = new_image_id
                    adjusted_ann['original_ann_id'] = ann['id']
                    patch_annotations.append(adjusted_ann)
                    new_ann_id += 1
            
            # Only add patch if it has annotations
            if patch_annotations:
                # Create new image entry
                new_img = {
                    'id': new_image_id,
                    'file_name': patch_info['patch_name'],
                    'width': patch_info['width'],
                    'height': patch_info['height'],
                    'original_image_id': img_id,
                    'original_file_name': img_info['file_name'],
                    'patch_id': patch_info['patch_id'],
                    'x_offset': patch_info['x_offset'],
                    'y_offset': patch_info['y_offset']
                }
                
                new_coco['images'].append(new_img)
                new_coco['annotations'].extend(patch_annotations)
                new_image_id += 1
            else:
                # Remove empty patch image
                patch_path = os.path.join(output_img_dir, patch_info['patch_name'])
                if os.path.exists(patch_path):
                    os.remove(patch_path)
    
    # Save new annotations
    output_ann_file = os.path.join(output_root, split, f'{split}.json')
    with open(output_ann_file, 'w') as f:
        json.dump(new_coco, f, indent=2)
    
    print(f"{split.upper()} split completed")
    print(f"Original images:      {len(coco_data['images']):4d}")
    print(f"Patched images:       {len(new_coco['images']):4d}")
    print(f"Original annotations: {len(coco_data['annotations']):4d}")
    print(f"Patched annotations:  {len(new_coco['annotations']):4d}")

def main():
    parser = argparse.ArgumentParser(description='Prepare patched dataset for dataset paper')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory of COCO format dataset')
    parser.add_argument('--output_root', type=str, required=True,
                      help='Output directory for patched dataset')
    parser.add_argument('--patch_size', type=int, default=341,
                      help='Size of each patch (default: 341)')
    parser.add_argument('--overlap', type=int, default=10,
                      help='Overlap between patches (default: 10)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                      help='Splits to process (default: train val test)')
    
    args = parser.parse_args()
    
    print("Patch-based dataset preparation")
    print(f"Patch size: {args.patch_size}x{args.patch_size}, Overlap: {args.overlap}px")
    
    for split in args.splits:
        create_patched_dataset(args.data_root, args.output_root, split, 
                             args.patch_size, args.overlap)
    
    print(f"Dataset created at: {args.output_root}")

if __name__ == '__main__':
    main()

