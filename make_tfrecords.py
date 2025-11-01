#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create TFRecords from rain/clean image pairs
Compatible with attentive-gan-derainnet data format
"""
import os
import os.path as ops
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import config to get image size settings
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config import global_config
CFG = global_config.cfg


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def morph_process(image):
    """
    Image morphological processing (same as original)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel)
    return open_image


def create_tfrecords(dataset_dir, output_dir):
    """
    Create TFRecords from image pairs using RAW BYTES format
    (matching the original write_example_tfrecords format)
    
    Args:
        dataset_dir: Directory containing train/val folders with rainy/clean subfolders
        output_dir: Directory to save tfrecords
    """
    
    if not ops.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get target size from config
    target_height = CFG.TRAIN.IMG_HEIGHT
    target_width = CFG.TRAIN.IMG_WIDTH
    
    print(f"\n[INFO] Target image size: {target_width}x{target_height}")
    print(f"[INFO] This matches the configuration in global_config.py\n")
    
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} dataset...")
        print(f"{'='*60}")
        
        rainy_dir = ops.join(dataset_dir, split, 'rainy')
        clean_dir = ops.join(dataset_dir, split, 'clean')
        
        if not ops.exists(rainy_dir):
            print(f"[WARNING] {rainy_dir} does not exist, skipping...")
            continue
        if not ops.exists(clean_dir):
            print(f"[WARNING] {clean_dir} does not exist, skipping...")
            continue
        
        # Get all rainy images
        rainy_images = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            import glob
            rainy_images.extend(glob.glob(ops.join(rainy_dir, ext)))
        
        rainy_images = sorted(rainy_images)
        
        if len(rainy_images) == 0:
            print(f"[ERROR] No images found in {rainy_dir}")
            continue
        
        print(f"Found {len(rainy_images)} rainy images")
        
        # Create TFRecord writer
        tfrecord_path = ops.join(output_dir, f'{split}.tfrecords')
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        
        success_count = 0
        error_count = 0
        
        # Process each image pair
        for rainy_path in tqdm(rainy_images, desc=f"Creating {split} tfrecords"):
            # Get corresponding clean image
            basename = ops.basename(rainy_path)
            clean_path = ops.join(clean_dir, basename)
            
            # Try different extensions if not found
            if not ops.exists(clean_path):
                name_without_ext = ops.splitext(basename)[0]
                found = False
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    clean_path_try = ops.join(clean_dir, name_without_ext + ext)
                    if ops.exists(clean_path_try):
                        clean_path = clean_path_try
                        found = True
                        break
                
                if not found:
                    print(f"\n[WARNING] Clean image not found for: {basename}")
                    error_count += 1
                    continue
            
            # Read images
            try:
                rain_image = cv2.imread(rainy_path, cv2.IMREAD_COLOR)
                clean_image = cv2.imread(clean_path, cv2.IMREAD_COLOR)
                
                if rain_image is None:
                    print(f"\n[ERROR] Failed to read: {rainy_path}")
                    error_count += 1
                    continue
                
                if clean_image is None:
                    print(f"\n[ERROR] Failed to read: {clean_path}")
                    error_count += 1
                    continue
                
                # Resize images to target size (matching original code)
                if rain_image.shape != (target_height, target_width, 3):
                    rain_image = cv2.resize(
                        rain_image,
                        dsize=(target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                if clean_image.shape != (target_height, target_width, 3):
                    clean_image = cv2.resize(
                        clean_image,
                        dsize=(target_width, target_height),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Convert to RAW BYTES (using .tostring() like original)
                rain_image_raw = rain_image.tostring()
                clean_image_raw = clean_image.tostring()
                
                # Create mask image (same as original)
                diff_image = np.abs(np.array(rain_image, np.float32) - np.array(clean_image, np.float32))
                diff_image = diff_image.sum(axis=2)
                
                mask_image = np.zeros(diff_image.shape, np.float32)
                mask_image[np.where(diff_image >= 35)] = 1.
                mask_image = morph_process(mask_image)
                mask_image_raw = mask_image.tostring()
                
                # Create feature (exact format as original)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'rain_image_raw': _bytes_feature(rain_image_raw),
                            'clean_image_raw': _bytes_feature(clean_image_raw),
                            'mask_image_raw': _bytes_feature(mask_image_raw)
                        }
                    )
                )
                
                writer.write(example.SerializeToString())
                success_count += 1
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {basename}: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
        
        writer.close()
        
        print(f"\n[DONE] {split.upper()} TFRecords created!")
        print(f"  Success: {success_count} pairs")
        print(f"  Errors: {error_count} pairs")
        print(f"  Saved to: {tfrecord_path}")
        print(f"  Image size: {target_width}x{target_height}")
        print(f"{'='*60}\n")


def init_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create TFRecords from image pairs')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Dataset directory containing train/val folders')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for tfrecords (default: dataset_dir/tfrecords)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    
    # Set default output dir
    if args.output_dir is None:
        args.output_dir = ops.join(args.dataset_dir, 'tfrecords')
    
    print(f"\n{'='*60}")
    print(f"TFRecords Creator for Deraining Dataset")
    print(f"{'='*60}")
    print(f"Input directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    create_tfrecords(args.dataset_dir, args.output_dir)
    
    print("\n✅ ALL DONE! You can now start training.")
    print(f"Run: python tools/train_model.py --dataset_dir \"{args.dataset_dir}\"")