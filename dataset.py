import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import functional as F

class JointTransform:
    def __call__(self, image_mask):
        if random.random() > 0.5:
            image_mask = F.hflip(image_mask)
        if random.random() > 0.5:
            image_mask = F.vflip(image_mask)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image_mask = F.rotate(image_mask, angle, expand=False)
        return image_mask


class FundusVesselDataset(Dataset):
    """
    Dataset class for eye fundus vessel segmentation using a few-shot learning approach.
    Assumes images are pre-cropped and enhanced during dataset preparation.
    """

    def __init__(self, base_dir, mode='train', transforms=None, n_shots=1, min_fg=100):
        """
        Args:
            base_dir: Directory containing the dataset.
            mode: 'train', 'val', or 'test'.
            transforms: Data augmentation transforms (applied only during training).
            n_shots: Number of support images per episode.
            min_fg: Minimum number of foreground pixels (not used in this simplified version).
        """
        super(FundusVesselDataset, self).__init__()
        self.base_dir = base_dir
        self.mode = mode
        self.transforms = transforms
        self.n_shots = n_shots
        self.min_fg = min_fg

        # List image files (TIF format)
        self.img_paths = sorted(
            glob.glob(os.path.join(base_dir, mode, "images", "*.tif")))

        # Find corresponding mask files (PNG format) with matching base names
        self.mask_paths = []
        for img_path in self.img_paths:
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]

            # Try potential mask filenames
            mask_found = False
            for mask_name in [f"{img_basename}.png", f"{img_basename}_mask.png"]:
                mask_path = os.path.join(base_dir, mode, "masks", mask_name)
                if os.path.exists(mask_path):
                    self.mask_paths.append(mask_path)
                    mask_found = True
                    break

            if not mask_found:
                # If no PNG mask found, try TIF masks as a fallback
                for mask_name in [f"{img_basename}.tif", f"{img_basename}_mask.tif"]:
                    mask_path = os.path.join(
                        base_dir, mode, "masks", mask_name)
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                        mask_found = True
                        break

            if not mask_found:
                raise ValueError(
                    f"No mask found for image {img_filename} in {os.path.join(base_dir, mode, 'masks')}")

        # Ensure we found masks for all images
        assert len(self.img_paths) == len(
            self.mask_paths), f"Number of images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) don't match"

        # Create few-shot episodes
        self._create_episode_lists()

        print(f"Loaded {len(self.img_paths)} image-mask pairs for {mode} set")

    def _create_episode_lists(self):
        """Create lists of support and query image indices for few-shot episodes."""
        n_images = len(self.img_paths)
        indices = list(range(n_images))

        self.episodes = []
        if self.mode == 'train':
            # For training, create random episodes.
            for _ in range(1000):  # Adjust the number of episodes as needed.
                random.shuffle(indices)
                support_indices = indices[:self.n_shots]
                query_indices = indices[self.n_shots:self.n_shots+1]
                self.episodes.append({
                    'support': support_indices,
                    'query': query_indices
                })
        else:
            # For validation/testing, create fixed episodes.
            step_size = self.n_shots + 1
            for i in range(0, n_images - step_size + 1, step_size):
                support_indices = indices[i:i+self.n_shots]
                query_indices = [indices[i+self.n_shots]]
                self.episodes.append({
                    'support': support_indices,
                    'query': query_indices
                })

    def __len__(self):
        # This is crucial for DataLoader to know the number of samples
        return len(self.episodes)

    def normalize_image(self, img):
        """Normalize image to zero mean and unit variance."""
        return (img - img.mean()) / (img.std() + 1e-5)

    def preprocess_image(self, img_path):
        """Load and preprocess an image without additional enhancement or resizing."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Direct conversion assuming images are already preprocessed.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
        #img = self.normalize_image(img)
        return img

    def preprocess_mask(self, mask_path):
        """Load and preprocess a mask without resizing."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask = (mask > 127).astype(np.float32)  # Binarize mask
        mask = torch.from_numpy(mask).float()
        return mask

    def get_fg_bg_masks(self, mask):
        """Return a dictionary containing foreground and background masks."""
        fg_mask = mask.clone()
        bg_mask = 1.0 - mask
        return {'fg_mask': fg_mask, 'bg_mask': bg_mask}

    def __getitem__(self, idx):
        """Return a simplified episode with support and query images/masks."""
        episode = self.episodes[idx]

        # Process support images and masks.
        support_images = []
        support_fg_masks = []
        support_bg_masks = []

        for support_idx in episode['support']:
            img = self.preprocess_image(self.img_paths[support_idx])
            mask = self.preprocess_mask(self.mask_paths[support_idx])
            if self.transforms is not None and self.mode == 'train':
                combined = torch.cat([img, mask.unsqueeze(0)], dim=0)
                combined = self.transforms(combined)
                img = combined[:3]
                mask = combined[3]
            support_images.append(img)
            masks = self.get_fg_bg_masks(mask)
            support_fg_masks.append(masks['fg_mask'])
            support_bg_masks.append(masks['bg_mask'])

        # Process query images and masks.
        query_images = []
        query_labels = []
        for query_idx in episode['query']:
            img = self.preprocess_image(self.img_paths[query_idx])
            mask = self.preprocess_mask(self.mask_paths[query_idx])
            query_images.append(img)
            query_labels.append(mask)

        return {
            # [Shot x 3 x H x W]
            'support_images':  torch.stack(support_images),
            # [Shot x H x W]
            'support_fg_masks':  torch.stack(support_fg_masks),
            # [Shot x H x W]
            'support_bg_masks':  torch.stack(support_bg_masks),
            # [Num_queries x 3 x H x W]
            'query_images':  torch.stack(query_images),
            # [Num_queries x H x W]
            'query_labels':  torch.stack(query_labels)
        }


def get_transforms():
    """Return an instance of JointTransform for data augmentation."""
    return JointTransform()


def get_dataloader(base_dir, mode='train', batch_size=1, n_shots=1, num_workers=4):
    """Create a dataloader for the fundus vessel dataset."""
    # transform = get_transforms() if mode == 'train' else None
    transform = None
    dataset = FundusVesselDataset(
        base_dir=base_dir,
        mode=mode,
        transforms=transform,
        n_shots=n_shots
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')
    )
    return dataloader, dataset
