import torch
import matplotlib.pyplot as plt
import numpy as np
from model import FewShotSeg
from dataset import get_dataloader
import warnings
warnings.filterwarnings("ignore", message="xFormers is available")

def visualize_sample(query_img, gt_mask, pred_mask):
    # Remove extra batch dimensions if present
    if query_img.dim() == 4 and query_img.shape[0] == 1:
        query_img = query_img.squeeze(0)
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask.squeeze(0)
    elif gt_mask.dim() == 4 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask.squeeze(0)
    if pred_mask.dim() == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    elif pred_mask.dim() == 4 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)

    # Convert tensors to numpy arrays
    query_np = query_img.cpu().permute(1, 2, 0).numpy()  # shape: (H, W, C)
    gt_np = gt_mask.cpu().numpy()  # expecting shape: (H, W)
    pred_np = pred_mask.cpu().numpy()  # expecting shape: (H, W)

    # If the mask arrays still have a singleton dimension on axis 0, squeeze it
    if gt_np.ndim == 3 and gt_np.shape[0] == 1:
        gt_np = gt_np.squeeze(0)
    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = pred_np.squeeze(0)

    # Plot the results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(query_np.astype(np.uint8) if query_np.max() > 1 else query_np)
    axs[0].set_title("Query Image")
    axs[0].axis("off")

    axs[1].imshow(gt_np, cmap="gray")
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_np, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = {
        'align': False,
        'which_model': "dinov2_vits",
        'proto_grid_size': 8,
        'feature_hw': [224 // 16, 224 // 16],
        'use_coco_init': False
    }
    
    checkpoint_path = "logs/run_20250408_152425/checkpoints/final_model.pth"
    model = FewShotSeg(
        image_size=224,
        pretrained_path=checkpoint_path,
        cfg=model_config
    )
    model.to(device)
    model.eval()

    test_loader, _ = get_dataloader(
        base_dir="processed",
        mode='test',
        batch_size=1,
        n_shots=2,
        num_workers=4  # Adjust if necessary; try 0 if problems persist
    )

    with torch.no_grad():
        for batch in test_loader:
            support_images = batch['support_images'].to(device)
            support_fg_mask = batch['support_fg_masks'].to(device)
            support_bg_mask = batch['support_bg_masks'].to(device)
            query_images = batch['query_images'].to(device)
            query_labels = batch['query_labels'].to(device)

            if support_fg_mask.dim() == 4:
                support_fg_mask = support_fg_mask.unsqueeze(2)
            if support_bg_mask.dim() == 4:
                support_bg_mask = support_bg_mask.unsqueeze(2)

            outputs, _, _, _, _, _, _ = model(
                support_images, support_fg_mask, support_bg_mask, query_images,
                isval=True, val_wsize=1
            )
            predicted_mask = outputs.argmax(dim=1)

            query_image_sample = query_images[0]
            ground_truth_sample = query_labels[0]
            predicted_mask_sample = predicted_mask[0]

            visualize_sample(query_image_sample, ground_truth_sample, predicted_mask_sample)
            # break

if __name__ == '__main__':
    main()
