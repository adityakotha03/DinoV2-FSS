import numpy as np
import torch
import random
import cv2
import os
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(object):
    """Compute evaluation metrics for segmentation"""
    def __init__(self, max_label=1, n_runs=1):
        self.labels = list(range(max_label + 1))  # All class labels
        self.n_runs = n_runs

        # Initialize tracking lists
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]
        self.tn_lst = [[] for _ in range(self.n_runs)]

    def reset(self):
        """Reset accumulated evaluation"""
        self.tp_lst = [[] for _ in range(self.n_runs)]
        self.fp_lst = [[] for _ in range(self.n_runs)]
        self.fn_lst = [[] for _ in range(self.n_runs)]
        self.tn_lst = [[] for _ in range(self.n_runs)]

    def record(self, pred, target, labels=None, n_run=0):
        """
        Record evaluation results for each sample and class
        
        Args:
            pred: Predicted segmentation mask
            target: Ground truth mask
            labels: Specific labels to evaluate
            n_run: Index of the current run
        """
        assert pred.shape == target.shape, "Prediction and target shapes don't match"
        
        # Default to all labels if not specified
        if labels is None:
            labels = self.labels
        
        # Arrays to store statistics for each class
        tp_arr = np.zeros(len(self.labels))
        fp_arr = np.zeros(len(self.labels))
        fn_arr = np.zeros(len(self.labels))
        tn_arr = np.zeros(len(self.labels))
        
        # Calculate metrics for each class
        for j, label in enumerate(labels):
            # True positives: pred == label and target == label
            tp = np.logical_and(pred == label, target == label).sum()
            # False positives: pred == label and target != label
            fp = np.logical_and(pred == label, target != label).sum()
            # False negatives: pred != label and target == label
            fn = np.logical_and(pred != label, target == label).sum()
            # True negatives: pred != label and target != label
            tn = np.logical_and(pred != label, target != label).sum()
            
            tp_arr[label] = tp
            fp_arr[label] = fp
            fn_arr[label] = fn
            tn_arr[label] = tn
        
        self.tp_lst[n_run].append(tp_arr)
        self.fp_lst[n_run].append(fp_arr)
        self.fn_lst[n_run].append(fn_arr)
        self.tn_lst[n_run].append(tn_arr)

    def get_mIoU(self, labels=None, n_run=None):
        """
        Compute mean IoU (Intersection over Union)
        
        Args:
            labels: Specific labels to evaluate
            n_run: Specific run to evaluate
        """
        if labels is None:
            labels = self.labels
        
        if n_run is None:
            # Compute for all runs and average
            ious = []
            for run in range(self.n_runs):
                tp_sum = np.sum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                fp_sum = np.sum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                fn_sum = np.sum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                
                # IoU = TP / (TP + FP + FN)
                iou_class = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-10)
                miou = np.mean(iou_class)
                ious.append((iou_class, miou))
            
            # Average across runs
            avg_iou_class = np.mean([iou[0] for iou in ious], axis=0)
            avg_miou = np.mean([iou[1] for iou in ious])
            
            return avg_iou_class, avg_miou
        else:
            # Compute for specific run
            tp_sum = np.sum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.sum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.sum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)
            
            # IoU = TP / (TP + FP + FN)
            iou_class = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-10)
            miou = np.mean(iou_class)
            
            return iou_class, miou

    def get_mDice(self, labels=None, n_run=None):
        """
        Compute mean Dice score
        
        Args:
            labels: Specific labels to evaluate
            n_run: Specific run to evaluate
        """
        if labels is None:
            labels = self.labels
        
        if n_run is None:
            # Compute for all runs and average
            dices = []
            for run in range(self.n_runs):
                tp_sum = np.sum(np.vstack(self.tp_lst[run]), axis=0).take(labels)
                fp_sum = np.sum(np.vstack(self.fp_lst[run]), axis=0).take(labels)
                fn_sum = np.sum(np.vstack(self.fn_lst[run]), axis=0).take(labels)
                
                # Dice = 2*TP / (2*TP + FP + FN)
                dice_class = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-10)
                mdice = np.mean(dice_class)
                dices.append((dice_class, mdice))
            
            # Average across runs
            avg_dice_class = np.mean([dice[0] for dice in dices], axis=0)
            avg_mdice = np.mean([dice[1] for dice in dices])
            
            return avg_dice_class, avg_mdice
        else:
            # Compute for specific run
            tp_sum = np.sum(np.vstack(self.tp_lst[n_run]), axis=0).take(labels)
            fp_sum = np.sum(np.vstack(self.fp_lst[n_run]), axis=0).take(labels)
            fn_sum = np.sum(np.vstack(self.fn_lst[n_run]), axis=0).take(labels)
            
            # Dice = 2*TP / (2*TP + FP + FN)
            dice_class = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-10)
            mdice = np.mean(dice_class)
            
            return dice_class, mdice

def visualize_results(image, mask, prediction, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        image: Input image (3, H, W)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W)
        save_path: Path to save visualization
    """
    # Convert image to numpy and normalize
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:  # Change from (3, H, W) to (H, W, 3)
        image = np.transpose(image, (1, 2, 0))
    
    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Create RGB visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(image)
    mask_vis = np.zeros_like(image)
    mask_vis[:, :, 1] = mask  # Green channel for ground truth
    axes[1].imshow(mask_vis, alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(image)
    pred_vis = np.zeros_like(image)
    pred_vis[:, :, 0] = prediction  # Red channel for prediction
    axes[2].imshow(pred_vis, alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_colored_mask(mask, save_path):
    """
    Save a colored segmentation mask
    
    Args:
        mask: Binary segmentation mask
        save_path: Path to save the colored mask
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 255, 255]  # White for vessels
    
    cv2.imwrite(save_path, colored_mask)