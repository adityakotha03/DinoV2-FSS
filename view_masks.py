import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_dataloader
from model import FewShotSeg

if __name__ == "__main__":
    base_dir = "processed"
    dataloader, dataset = get_dataloader(base_dir=base_dir, mode='train', batch_size=1, n_shots=1)
    batch = next(iter(dataloader))

    # support_images shape: [B, n_shot, 3, H, W]
    supp_imgs = batch['support_images']
    fore_mask = batch['support_fg_masks'].unsqueeze(2)  # becomes [B, n_shot, 1, H, W]
    back_mask = batch['support_bg_masks'].unsqueeze(2)  # becomes [B, n_shot, 1, H, W]


    qry_imgs = batch['query_images']  # e.g. [B, num_queries, 3, H, W]
    qry_imgs = qry_imgs[:, 0]  # now [B, 3, H, W]

    # Initialize your few-shot segmentation model.
    model = FewShotSeg(image_size=224, pretrained_path=None)
    model.eval()

    B, n_shot, C, H, W = supp_imgs.shape

    # Flatten support images and masks.
    supp_imgs_flat = supp_imgs.view(B * n_shot, C, H, W)
    fore_mask_flat = fore_mask.view(B * n_shot, 1, H, W)
    back_mask_flat = back_mask.view(B * n_shot, 1, H, W)

    # Extract features from support images using the same encoder.
    supp_feat = model.get_features(supp_imgs_flat)
    print("Support features shape:", supp_feat.shape)
    _, _, h_feat, w_feat = supp_feat.shape

    # Resize masks to match feature map size.
    fore_mask_resized = F.interpolate(fore_mask_flat, size=(h_feat, w_feat), mode='nearest')
    back_mask_resized = F.interpolate(back_mask_flat, size=(h_feat, w_feat), mode='nearest')

    # Reshape them back to [B, n_shot, 1, h_feat, w_feat]
    fore_mask_resized = fore_mask_resized.view(B, n_shot, 1, h_feat, w_feat)
    back_mask_resized = back_mask_resized.view(B, n_shot, 1, h_feat, w_feat)
    print(fore_mask_resized.shape)
    print(back_mask_resized.shape)

    # --- Visualization ---
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 5))
    ax0.imshow(fore_mask_flat[0, 0].cpu().numpy(), cmap='gray')
    ax0.set_title("Original Foreground Mask")
    ax0.axis('off')
    ax1.imshow(fore_mask_resized[0, 0, 0].cpu().numpy(), cmap='gray')
    ax1.set_title("Resized Foreground Mask")
    ax1.axis('off')
    ax2.imshow(back_mask_resized[0, 0, 0].cpu().numpy(), cmap='gray')
    ax2.set_title("Resized Background Mask")
    ax2.axis('off')
    feature_channel = supp_feat[0, 0].cpu().detach().numpy()  # shape: [h_feat, w_feat]
    ax3.imshow(feature_channel, cmap='viridis')
    ax3.set_title("Feature Channel 0")
    ax3.axis('off')
    plt.show()