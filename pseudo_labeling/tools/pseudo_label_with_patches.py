#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from pycocotools import mask as maskUtils
import warnings
warnings.filterwarnings('ignore')

CATEGORY_NAMES = ['pore area', 'guard cell area', 'complex area']
CATEGORY_NAME_SET = {name.lower() for name in CATEGORY_NAMES}


def parse_class_thresholds(threshold_args):
    """Parse class-specific threshold arguments into a dictionary."""

    thresholds = {}
    if not threshold_args:
        return thresholds

    for item in threshold_args:
        if '=' not in item:
            continue
        name, value = item.split('=', 1)
        name = name.strip().lower()
        if name not in CATEGORY_NAME_SET:
            continue
        try:
            thresholds[name] = float(value)
        except ValueError:
            continue

    return thresholds


def get_threshold_for_label(label_idx, thresholds_by_name, default_threshold):
    class_name = CATEGORY_NAMES[label_idx].lower() if label_idx < len(CATEGORY_NAMES) else None
    if class_name and class_name in thresholds_by_name:
        return thresholds_by_name[class_name]
    return default_threshold

CATEGORY_NAMES = ['pore', 'guard_cell', 'complex', 'stomata']
CATEGORY_NAME_SET = set(CATEGORY_NAMES)

def crop_image_to_patches(image_path, patch_size=341, overlap=10):
    """
    Crop image into 3x3 grid patches
    Returns list of (patch_array, x_offset, y_offset, width, height)
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    
    stride = patch_size - overlap
    patches = []
    
    # 3x3 grid
    for row in range(3):
        for col in range(3):
            x_offset = col * stride
            y_offset = row * stride
            
            # Ensure we don't go out of bounds
            x_end = min(x_offset + patch_size, img_width)
            y_end = min(y_offset + patch_size, img_height)
            
            # Adjust start if needed
            if x_end - x_offset < patch_size and x_offset > 0:
                x_offset = x_end - patch_size
            if y_end - y_offset < patch_size and y_offset > 0:
                y_offset = y_end - patch_size
            
            # Extract patch
            patch = img_array[y_offset:y_end, x_offset:x_end]
            
            patches.append({
                'patch': patch,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'width': x_end - x_offset,
                'height': y_end - y_offset
            })
    
    return patches, img_width, img_height

def adjust_detections_to_full_image(detections, x_offset, y_offset, full_width, full_height):
    """
    Adjust detection coordinates from patch to full image
    """
    adjusted = []
    
    for det in detections:
        # Adjust bbox
        bbox = det['bbox'].copy()
        bbox[0] += x_offset  # x1
        bbox[1] += y_offset  # y1
        bbox[2] += x_offset  # x2
        bbox[3] += y_offset  # y2
        
        # Adjust segmentation
        seg = det['segmentation'].copy()
        seg['counts'] = seg['counts'].encode('utf-8') if isinstance(seg['counts'], str) else seg['counts']
        
        # Decode, shift, and re-encode
        mask = maskUtils.decode(seg)
        
        # Create full image mask with consistent size
        full_mask = np.zeros((full_height, full_width), dtype=np.uint8)
        y_end = min(y_offset + mask.shape[0], full_height)
        x_end = min(x_offset + mask.shape[1], full_width)

        # Determine valid region within mask bounds
        mask_y_end = y_end - y_offset
        mask_x_end = x_end - x_offset

        if mask_y_end <= 0 or mask_x_end <= 0:
            continue

        full_mask[y_offset:y_end, x_offset:x_end] = mask[:mask_y_end, :mask_x_end]
        
        # Re-encode
        full_seg = maskUtils.encode(np.asfortranarray(full_mask))
        full_seg['counts'] = full_seg['counts'].decode('utf-8')
        
        adjusted.append({
            'bbox': bbox,
            'score': det['score'],
            'category_id': det['category_id'],
            'segmentation': full_seg
        })
    
    return adjusted

def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections at patch boundaries
    """
    if not detections:
        return []
    
    # Group by category
    detections_by_cat = {}
    for det in detections:
        cat_id = det['category_id']
        if cat_id not in detections_by_cat:
            detections_by_cat[cat_id] = []
        detections_by_cat[cat_id].append(det)
    
    # Apply NMS per category
    final_detections = []
    
    for cat_id, dets in detections_by_cat.items():
        # Sort by score
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while dets:
            # Keep highest scoring detection
            best = dets.pop(0)
            keep.append(best)
            
            # Calculate IoU with remaining detections
            if not dets:
                break
            
            # Decode masks
            best_mask = maskUtils.decode({
                'size': best['segmentation']['size'],
                'counts': best['segmentation']['counts'].encode('utf-8') 
                         if isinstance(best['segmentation']['counts'], str) 
                         else best['segmentation']['counts']
            })
            
            remaining = []
            for det in dets:
                det_mask = maskUtils.decode({
                    'size': det['segmentation']['size'],
                    'counts': det['segmentation']['counts'].encode('utf-8') 
                             if isinstance(det['segmentation']['counts'], str) 
                             else det['segmentation']['counts']
                })
                
                # Calculate IoU
                intersection = np.logical_and(best_mask, det_mask).sum()
                union = np.logical_or(best_mask, det_mask).sum()
                iou = intersection / union if union > 0 else 0
                
                # Keep if IoU is below threshold
                if iou < iou_threshold:
                    remaining.append(det)
            
            dets = remaining
        
        final_detections.extend(keep)
    
    return final_detections

# Legacy path left for compatibility (not used in patch-only workflow)
def inference_on_patches(model, image_path, confidence_threshold=0.5, 
                        patch_size=341, overlap=10):
    patches, img_width, img_height = crop_image_to_patches(image_path, patch_size, overlap)
    all_detections = []
    for patch_info in patches:
        patch = patch_info['patch']
        result = inference_detector(model, patch)
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        masks = pred_instances.masks.cpu().numpy()
        patch_detections = []
        for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
            if score < confidence_threshold:
                continue
            mask_uint8 = mask.astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(mask_uint8))
            rle['counts'] = rle['counts'].decode('utf-8')
            patch_detections.append({
                'bbox': bbox.tolist(),
                'score': float(score),
                'category_id': int(label) + 1,
                'segmentation': rle
            })
        adjusted_detections = adjust_detections_to_full_image(
            patch_detections,
            patch_info['x_offset'],
            patch_info['y_offset'],
            img_width,
            img_height
        )
        all_detections.extend(adjusted_detections)
    final_detections = apply_nms(all_detections, iou_threshold=0.5)
    return final_detections, img_width, img_height


def inference_on_single_patch(model, patch_path, thresholds_by_name, default_threshold=0.5):
    """Run inference on a single patch path and return detections plus data sample."""

    result = inference_detector(model, patch_path)
    pred_instances = result.pred_instances

    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy()

    detections = []

    for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
        class_threshold = get_threshold_for_label(label, thresholds_by_name, default_threshold)
        if class_threshold is not None and score < class_threshold:
            continue

        mask_uint8 = mask.astype(np.uint8)
        rle = maskUtils.encode(np.asfortranarray(mask_uint8))
        rle['counts'] = rle['counts'].decode('utf-8')

        detections.append({
            'bbox': bbox.tolist(),
            'score': float(score),
            'category_id': int(label) + 1,
            'segmentation': rle
        })

    return detections, result


def save_data_sample_overlay(patch_path, detections, data_sample, overlay_path):
    """Save overlay using MMDetection's visualization utilities."""

    if data_sample is None:
        # Fallback to manual overlay if data sample not provided (shouldn't happen in patch mode)
        base_image = Image.open(patch_path).convert('RGB')
        overlay = np.array(base_image, dtype=np.uint8)
        alpha = 0.5
        color_map = {
            1: np.array([255, 0, 0], dtype=np.uint8),       # pore - red
            2: np.array([0, 255, 0], dtype=np.uint8),       # guard_cell - green
            3: np.array([0, 0, 255], dtype=np.uint8),       # complex - blue
            4: np.array([255, 255, 0], dtype=np.uint8),     # stomata - yellow
        }

        for det in detections:
            seg = det['segmentation']
            if not seg:
                continue

            if isinstance(seg, dict):
                seg_rle = seg.copy()
                seg_rle['counts'] = (
                    seg_rle['counts'].encode('utf-8') if isinstance(seg_rle['counts'], str) else seg_rle['counts']
                )
                mask = maskUtils.decode(seg_rle).astype(bool)
            else:
                continue

            color = color_map.get(det['category_id'], np.array([255, 255, 255], dtype=np.uint8))
            overlay[mask] = (alpha * color + (1 - alpha) * overlay[mask]).astype(np.uint8)

        Image.fromarray(overlay).save(overlay_path)
        return

    data_sample_copy = data_sample.clone()

    try:
        visualizer = VISUALIZERS.build(dict(type='DetLocalVisualizer'))
        visualizer.dataset_meta = {'classes': CATEGORY_NAMES}

        visualizer.add_datasample(
            name='pseudo_patch',
            image=Image.open(patch_path).convert('RGB'),
            data_sample=data_sample_copy,
            out_file=overlay_path,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=0.0,
            show=False
        )
    except Exception:
        base_image = Image.open(patch_path).convert('RGB')
        overlay = np.array(base_image, dtype=np.uint8)
        alpha = 0.5
        color_map = {
            1: np.array([255, 0, 0], dtype=np.uint8),
            2: np.array([0, 255, 0], dtype=np.uint8),
            3: np.array([0, 0, 255], dtype=np.uint8),
            4: np.array([255, 255, 0], dtype=np.uint8),
        }

        for det in detections:
            seg = det['segmentation']
            if not seg:
                continue

            if isinstance(seg, dict):
                seg_rle = seg.copy()
                seg_rle['counts'] = (
                    seg_rle['counts'].encode('utf-8') if isinstance(seg_rle['counts'], str) else seg_rle['counts']
                )
                mask = maskUtils.decode(seg_rle).astype(bool)
            else:
                continue

            color = color_map.get(det['category_id'], np.array([255, 255, 255], dtype=np.uint8))
            overlay[mask] = (alpha * color + (1 - alpha) * overlay[mask]).astype(np.uint8)

        Image.fromarray(overlay).save(overlay_path)


def inference_on_precomputed_patches(model, img_info, patch_image_dir,
                                     img_width, img_height, thresholds_by_name, default_threshold=0.5):
    """Run inference on precomputed patches and combine results."""

    all_detections = []

    for patch_info in img_info['patches']:
        patch_path = os.path.join(patch_image_dir, patch_info['patch_name'])
        if not os.path.exists(patch_path):
            continue

        detections, _ = inference_on_single_patch(
            model,
            patch_path,
            thresholds_by_name,
            default_threshold
        )

        patch_detections = detections

        adjusted_detections = adjust_detections_to_full_image(
            patch_detections,
            patch_info['x_offset'],
            patch_info['y_offset'],
            img_width,
            img_height
        )

        all_detections.extend(adjusted_detections)

    final_detections = apply_nms(all_detections, iou_threshold=0.5)

    return final_detections

def convert_to_coco_format(detections, img_id, img_width, img_height, img_name):
    """
    Convert detections to COCO format annotations
    """
    annotations = []
    
    for ann_id, det in enumerate(detections, start=1):
        # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = det['bbox']
        bbox_coco = [x1, y1, x2 - x1, y2 - y1]
        
        # Decode mask to get area
        rle = det['segmentation']
        rle['counts'] = rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts']
        mask = maskUtils.decode(rle)
        area = int(mask.sum())
        
        # Re-encode with string counts for JSON
        rle['counts'] = rle['counts'].decode('utf-8')
        
        # Convert RLE to polygon (optional, for visualization)
        # For now, keep RLE format
        
        annotation = {
            'id': ann_id,
            'image_id': img_id,
            'category_id': det['category_id'],
            'bbox': bbox_coco,
            'area': area,
            'segmentation': rle,
            'iscrowd': 0,
            'score': det['score']  # Keep confidence score
        }
        
        annotations.append(annotation)
    
    return annotations

def main():
    parser = argparse.ArgumentParser(description='Pseudo-label unlabeled images using patch-based inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--unlabeled_dir', type=str, required=True,
                       help='Directory containing unlabeled images')
    parser.add_argument('--unlabeled_patches_dir', type=str,
                       help='Optional directory containing precomputed unlabeled patches')
    parser.add_argument('--patch_mode', action='store_true',
                       help='When set, perform inference directly on precomputed patches')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for pseudo-labels')
    parser.add_argument('--overlay_dir', type=str,
                       help='Optional directory to save overlay visualizations')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Default confidence threshold for pseudo-labels (fallback)')
    parser.add_argument('--patch_size', type=int, default=341,
                       help='Patch size (default: 341)')
    parser.add_argument('--overlap', type=int, default=10,
                       help='Patch overlap (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference')
    parser.add_argument('--class_thresholds', nargs='*', default=None,
                       help='Per-class thresholds, e.g., "pore area=0.5" "guard cell area=0.7"')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Patch-based pseudo-labeling with instance segmentation")
    print(f"Model config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Unlabeled data: {args.unlabeled_dir}")
    if args.unlabeled_patches_dir:
        print(f"Unlabeled patches: {args.unlabeled_patches_dir}")
    if args.patch_mode:
        print("Inference mode: precomputed patches")
    print(f"Default threshold: {args.confidence_threshold}")
    thresholds_by_name = parse_class_thresholds(args.class_thresholds)
    if thresholds_by_name:
        print("Per-class thresholds:")
        for cls, thr in thresholds_by_name.items():
            print(f"  - {cls}: {thr}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}, Overlap: {args.overlap}px")
    
    register_all_modules()
    
    print("Loading model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("Model loaded successfully")
    
    # Determine if precomputed patches are provided
    use_precomputed_patches = bool(args.unlabeled_patches_dir)
    patch_metadata = None

    if args.patch_mode and not use_precomputed_patches:
        raise ValueError("--patch_mode requires --unlabeled_patches_dir")

    if use_precomputed_patches:
        metadata_path = os.path.join(args.unlabeled_patches_dir, 'patch_metadata.json')
        patch_image_dir = os.path.join(args.unlabeled_patches_dir, 'images')

        if not os.path.isdir(patch_image_dir):
            raise FileNotFoundError(
                f"Expected patch images directory not found: {patch_image_dir}"
            )

        original_images = None
        metadata_lookup = {}
        patch_dims = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                patch_metadata = json.load(f)
            original_images = patch_metadata.get('images', [])
            for img in original_images:
                for patch in img.get('patches', []):
                    metadata_lookup[patch['patch_name']] = {
                        'original_file_name': img.get('original_file_name'),
                        'x_offset': patch.get('x_offset'),
                        'y_offset': patch.get('y_offset'),
                        'patch_id': patch.get('patch_id')
                    }
                    patch_dims[patch['patch_name']] = (patch.get('width'), patch.get('height'))

        patch_files = sorted([
            f for f in os.listdir(patch_image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        image_files = None
        total_input_images = len(patch_files)
        print(f"Found {total_input_images} precomputed patch images")
    else:
        image_files = sorted([
            f for f in os.listdir(args.unlabeled_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        original_images = None
        total_input_images = len(image_files)
        print(f"Found {total_input_images} unlabeled images")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_img_dir = os.path.join(args.output_dir, 'train' if args.patch_mode else 'images')
    os.makedirs(output_img_dir, exist_ok=True)

    overlay_dir = None
    overlay_dir = None
    if args.overlay_dir:
        overlay_dir = os.path.join(args.overlay_dir, 'train' if args.patch_mode else 'images')
        os.makedirs(overlay_dir, exist_ok=True)

    # Initialize COCO format data
    coco_data = {
        'info': {
            'description': 'Pseudo-labeled Stomata Dataset (Patch-based)',
            'version': '1.0',
            'year': 2025
        },
        'licenses': [],
        'categories': [
            {'id': 1, 'name': 'pore area', 'supercategory': 'stomata'},
            {'id': 2, 'name': 'guard cell area', 'supercategory': 'stomata'},
            {'id': 3, 'name': 'complex area', 'supercategory': 'stomata'}
        ],
        'images': [],
        'annotations': []
    }

    total_annotations = 0
    ann_id_counter = 1

    if args.patch_mode:
        image_counter = 1
        for patch_name in tqdm(patch_files, desc="Generating pseudo-labels (patch mode)"):
            patch_path = os.path.join(patch_image_dir, patch_name)

            meta_dims = patch_dims.get(patch_name)
            if meta_dims:
                patch_width, patch_height = meta_dims
            else:
                with Image.open(patch_path) as img:
                    patch_width, patch_height = img.size

            detections, data_sample = inference_on_single_patch(
                model,
                patch_path,
                thresholds_by_name,
                args.confidence_threshold
            )

            # Keep ALL patches, even empty ones (to prevent false positives)
            if detections:
                annotations = convert_to_coco_format(
                    detections,
                    image_counter,
                    patch_width,
                    patch_height,
                    patch_name
                )
            else:
                # Empty patch - no annotations
                annotations = []

            patch_meta = metadata_lookup.get(patch_name, {})

            coco_data['images'].append({
                'id': image_counter,
                'file_name': patch_name,
                'width': patch_width,
                'height': patch_height,
                'original_file_name': patch_meta.get('original_file_name'),
                'x_offset': patch_meta.get('x_offset'),
                'y_offset': patch_meta.get('y_offset'),
                'patch_id': patch_meta.get('patch_id')
            })

            for ann in annotations:
                ann['id'] = ann_id_counter
                coco_data['annotations'].append(ann)
                ann_id_counter += 1

            total_annotations += len(annotations)

            import shutil
            shutil.copy(patch_path, os.path.join(output_img_dir, patch_name))

            if overlay_dir and detections:  # Only save overlay if there are detections
                overlay_path = os.path.join(overlay_dir, patch_name.replace('.jpg', '_overlay.jpg'))
                save_data_sample_overlay(patch_path, detections, data_sample, overlay_path)

            image_counter += 1
    else:
        if use_precomputed_patches:
            iterator = enumerate(original_images, start=1)
        else:
            iterator = enumerate(image_files, start=1)

        for img_id, img_entry in tqdm(iterator, desc="Generating pseudo-labels"):
            if use_precomputed_patches:
                img_info = img_entry
                img_name = img_info['original_file_name']
                img_path = os.path.join(args.unlabeled_dir, img_name)
                img_width = img_info['original_width']
                img_height = img_info['original_height']

                detections = inference_on_precomputed_patches(
                    model,
                    img_info,
                    patch_image_dir,
                    img_width,
                    img_height,
                    thresholds_by_name,
                    default_threshold=args.confidence_threshold
                )
            else:
                img_name = img_entry
                img_path = os.path.join(args.unlabeled_dir, img_name)

                detections, img_width, img_height = inference_on_patches(
                    model,
                    img_path,
                    confidence_threshold=args.confidence_threshold,
                    patch_size=args.patch_size,
                    overlap=args.overlap
                )

            if not detections:
                continue

            annotations = convert_to_coco_format(
                detections,
                img_id,
                img_width,
                img_height,
                img_name
            )

            coco_data['images'].append({
                'id': img_id,
                'file_name': img_name,
                'width': img_width,
                'height': img_height
            })

            for ann in annotations:
                ann['id'] = ann_id_counter
                coco_data['annotations'].append(ann)
                ann_id_counter += 1

            total_annotations += len(annotations)

            import shutil
            shutil.copy(img_path, os.path.join(output_img_dir, img_name))
            if overlay_dir:
                overlay_path = os.path.join(overlay_dir, img_name.replace('.jpg', '_overlay.jpg'))
                save_data_sample_overlay(img_path, detections, None, overlay_path)
    
    # Save COCO format JSON
    output_json = os.path.join(args.output_dir, 'pseudo_labels.json')
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Pseudo-labeling completed")
    processed_count = total_input_images if not args.patch_mode else total_input_images
    print(f"Images processed: {processed_count}")
    print(f"Images with labels: {len(coco_data['images'])}")
    print(f"Total annotations: {total_annotations}")
    avg_annotations = total_annotations / len(coco_data['images']) if coco_data['images'] else 0
    print(f"Avg annotations/img: {avg_annotations:.1f}")
    print(f"Output JSON: {output_json}")
    print(f"Output images: {output_img_dir}")
    if overlay_dir:
        print(f"Overlays: {overlay_dir}")

if __name__ == '__main__':
    main()

