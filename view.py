import matplotlib.pyplot as plt
from dataset import get_dataloader
import torch

if __name__ == "__main__":
    base_dir = "processed"
    dataloader, dataset = get_dataloader(base_dir=base_dir, mode='train', batch_size=1, n_shots=1)
    batch = next(iter(dataloader))

    # Extract components from the batch.
    support_images = batch['support_images']         # [n_shots x 3 x H x W]
    support_fg_masks = batch['support_fg_masks']         # [n_shots x H x W]
    support_bg_masks = batch['support_bg_masks']         # [n_shots x H x W]
    query_images = batch['query_images']               # [num_queries x 3 x H x W]
    query_labels = batch['query_labels']               # [num_queries x H x W]

    # Print out the shapes.
    print("Support images shape:", support_images.shape)
    print("Support FG masks shape:", support_fg_masks.shape)
    print("Support BG masks shape:", support_bg_masks.shape)
    print("Query images shape:", query_images.shape)
    print("Query masks shape:", query_labels.shape)

    # Visualize the first query image and its mask.
    query_img = query_images[0, 0].permute(1, 2, 0).cpu().numpy()
    query_mask = query_labels[0, 0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(query_img)
    ax1.set_title("Query Image")
    ax1.axis('off')
    ax2.imshow(query_mask, cmap='gray')
    ax2.set_title("Query Mask")
    ax2.axis('off')
    plt.show()

    # Visualize all support images and their corresponding foreground and background masks.
    n_shots = support_images.shape[1]
    for i in range(n_shots):
        supp_img = support_images[0, i].permute(1, 2, 0).cpu().numpy()  # Index batch 0 then shot i.
        supp_fg = support_fg_masks[0, i].cpu().numpy()
        supp_bg = support_bg_masks[0, i].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(supp_img)
        axs[0].set_title(f"Support Image {i+1}")
        axs[0].axis('off')
        axs[1].imshow(supp_fg, cmap='gray')
        axs[1].set_title(f"Support FG Mask {i+1}")
        axs[1].axis('off')
        axs[2].imshow(supp_bg, cmap='gray')
        axs[2].set_title(f"Support BG Mask {i+1}")
        axs[2].axis('off')
        plt.show()