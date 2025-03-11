import os
import shutil
import argparse
import random
import glob
from tqdm import tqdm
import cv2

from utils.resize import max_pooling_resize

def prepare_dataset(source_dir: str, target_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Prepare the fundus vessel dataset by organizing TIF images and PNG masks
    into train, val, and test sets
    """
    os.makedirs(target_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'masks'), exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(source_dir, 'images', '*.tif')))
    if not image_files:
        raise ValueError(f"No TIF image files found in {os.path.join(source_dir, 'images')}")
    
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_images = image_files[:n_train]
    val_images = image_files[n_train:n_train+n_val]
    test_images = image_files[n_train+n_val:]
    
    print(f"Total images: {n_total}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Testing images: {len(test_images)}")
    
    def copy_files(image_list, split):
        for img_path in tqdm(image_list, desc=f"Copying {split} set"):
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            mask_filename = f"{img_basename}.png"
            mask_path = os.path.join(source_dir, 'masks', mask_filename)
            if not os.path.exists(mask_path):
                mask_filename = f"{img_basename}_mask.png"
                mask_path = os.path.join(source_dir, 'masks', mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {img_filename}, skipping file")
                    continue
            shutil.copy(img_path, os.path.join(target_dir, split, 'images', img_filename))
            shutil.copy(mask_path, os.path.join(target_dir, split, 'masks', os.path.basename(mask_path)))
    
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    print("Dataset preparation completed!")

def enhance_images(dataset_dir: str, use_clahe: bool = True, resize=None):
     # Set up CLAHE if needed
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(dataset_dir, split, 'images')
        image_files = glob.glob(os.path.join(image_dir, '*.tif'))
        for img_path in tqdm(image_files, desc=f"Enhancing {split} images"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}, skipping")
                continue
            if use_clahe:
                g_channel = img[:, :, 1]
                img[:, :, 1] = clahe.apply(g_channel)
            if resize:
                img = cv2.resize(img, tuple(resize), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(img_path, img)
        if resize:
            mask_dir = os.path.join(dataset_dir, split, 'masks')
            mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
            for mask_path in tqdm(mask_files, desc=f"Resizing {split} masks"):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not read {mask_path}, skipping")
                    continue
                # mask = cv2.resize(mask, tuple(resize), interpolation=cv2.INTER_NEAREST)
                mask = max_pooling_resize(mask, tuple(resize))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(mask_path, mask)
    print("Image enhancement completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fundus vessel dataset")
    parser.add_argument("--source_dir", type=str, default='data', help="Directory containing original images and masks")
    parser.add_argument("--target_dir", type=str, default='processed', help="Directory to save organized dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of images for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of images for validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of images for test set")
    parser.add_argument("--enhance", action="store_true", default=True, help="Apply image enhancement after preparation")
    parser.add_argument("--use_clahe", action="store_true", default=True, help="Use CLAHE for contrast enhancement")
    parser.add_argument("--resize", type=int, nargs=2, default=[224, 224], help="Resize images to specified width and height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    random.seed(args.seed)
    prepare_dataset(source_dir=args.source_dir, target_dir=args.target_dir,
                    train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    if args.enhance:
        enhance_images(dataset_dir=args.target_dir, use_clahe=args.use_clahe, resize=args.resize)
