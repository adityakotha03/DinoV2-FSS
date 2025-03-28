import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse
from datetime import datetime

from model import FewShotSeg
from dataset import get_dataloader, get_transforms
from utils.utils import AverageMeter, Metric, set_seed
import warnings
warnings.filterwarnings("ignore", message="xFormers is available")


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def train(config):
    """Main training function"""
    set_seed(config.seed)
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Initialize model configuration
    model_config = {
        'align': config.align,
        'which_model': config.model_name,
        'proto_grid_size': config.proto_grid_size,
        'feature_hw': [config.image_size // 16, config.image_size // 16],
        'use_coco_init': config.use_coco_init
    }

    # Initialize model
    model = FewShotSeg(
        image_size=config.image_size,
        pretrained_path=config.reload_model_path,
        cfg=model_config
    )
    model = model.to(device)
    model.train()

    # Create data loaders (removed unsupported arguments)
    train_loader, _ = get_dataloader(
        base_dir=config.data_dir,
        mode='train',
        batch_size=config.batch_size,
        n_shots=config.n_shots,
        num_workers=config.num_workers
    )

    val_loader, _ = get_dataloader(
        base_dir=config.data_dir,
        mode='val',
        batch_size=1,
        n_shots=config.n_shots,
        num_workers=config.num_workers
    )

    # Set up optimizer
    if config.optim_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optim_type == 'adam':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    # Set up scheduler
    milestones = [int(config.n_steps * 0.7), int(config.n_steps * 0.9)]
    scheduler = MultiStepLR(
        optimizer, milestones=milestones, gamma=config.lr_gamma)

    # Set up loss function
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([0.05, 1.0]).to(device))

    # Training loop
    i_iter = 0
    loss_meter = AverageMeter()
    align_loss_meter = AverageMeter()
    best_dice = 0.0

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, sample_batched in enumerate(pbar):
            i_iter += 1

            # list of tensors / list of support img
            # print(sample_batched['support_images'].shape)
            # print(sample_batched['support_bg_masks'].shape)
            # print(sample_batched['support_fg_masks'].shape)
            # print(sample_batched['query_images'].shape)
            # print(sample_batched['query_labels'].shape)

            support_images = sample_batched['support_images'].to(device)
            support_fg_mask = torch.unsqueeze(
                sample_batched['support_fg_masks'], dim=2).to(device)
            support_bg_mask = torch.unsqueeze(
                sample_batched['support_bg_masks'], dim=2).to(device)
            query_images = sample_batched['query_images'].to(device)
            query_labels = torch.unsqueeze(
                sample_batched['query_labels'], dim=2).to(device)
            # print("==========")
            # print(support_images.shape)
            # print(support_fg_masks.shape)
            # print(support_bg_masks.shape)
            # print(query_images.shape)
            # print(query_labels.shape)

            # Forward pass
            optimizer.zero_grad()
            outputs, align_loss, _, _, _, _, _ = model(
                support_images, support_fg_mask, support_bg_mask, query_images
            )

            # Compute loss
            query_loss = criterion(outputs, query_labels)
            loss = query_loss + align_loss

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update meters
            loss_meter.update(query_loss.item())
            align_loss_meter.update(align_loss.item() if isinstance(
                align_loss, torch.Tensor) else align_loss)

            # Print progress
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'align_loss': f"{align_loss_meter.avg:.4f}"
            })

            # Validation
            if i_iter % config.val_interval == 0:
                dice_score = validate(model, val_loader, device, config)
                print(
                    f"Iteration {i_iter} | Validation Dice: {dice_score:.4f}")

                if dice_score > best_dice:
                    best_dice = dice_score
                    save_checkpoint(
                        model, optimizer, epoch,
                        os.path.join(log_dir, "checkpoints", f"best_model.pth")
                    )

            # Save periodic checkpoint
            if i_iter % config.save_interval == 0:
                save_checkpoint(
                    model, optimizer, epoch,
                    os.path.join(log_dir, "checkpoints", f"iter_{i_iter}.pth")
                )

            if i_iter >= config.n_steps:
                break

        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            os.path.join(log_dir, "checkpoints", f"epoch_{epoch+1}.pth")
        )

        if i_iter >= config.n_steps:
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch,
        os.path.join(log_dir, "checkpoints", "final_model.pth")
    )

    print(f"Training completed! Best validation Dice: {best_dice:.4f}")


def validate(model, val_loader, device, config):
    """Validate the model"""
    model.eval()
    metric = Metric(max_label=1)

    with torch.no_grad():
        for sample_batched in tqdm(val_loader, desc="Validating"):
            # Prepare data
            support_images = [[shot.to(device) for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot['fg_mask'].float().to(device) for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot['bg_mask'].float().to(device) for shot in way]
                               for way in sample_batched['support_mask']]
            query_images = [query_image.to(
                device) for query_image in sample_batched['query_images']]
            query_labels = torch.cat([query_label.to(device)
                                      for query_label in sample_batched['query_labels']], dim=0)

            # Forward pass
            outputs, _, _, _, _, _, _ = model(
                support_images, support_fg_mask, support_bg_mask, query_images,
                isval=True, val_wsize=config.val_wsize
            )

            # Get predictions
            query_pred = outputs.argmax(dim=1).cpu().numpy()
            query_labels = query_labels.cpu().numpy()

            # Update metrics
            for pred, label in zip(query_pred, query_labels):
                metric.record(pred, label, labels=[1])

    # Calculate Dice score
    dice_class, dice_mean = metric.get_mDice(labels=[1])
    model.train()

    return dice_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ALPNet for fundus vessel segmentation")

    # Dataset and model parameters
    parser.add_argument("--data_dir", type=str, default='processed',
                        help="Directory containing the dataset")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--model_name", type=str, default="dinov2_vits",
                        help="Model name: dinov2_vits, dinov2_vitb, dinov2_vitl")
    parser.add_argument("--reload_model_path", type=str, default="",
                        help="Path to pretrained model weights")
    parser.add_argument("--use_coco_init", action="store_true",
                        help="Use COCO pretrained weights for initialization")
    parser.add_argument("--use_clahe", action="store_true",
                        help="Use CLAHE for contrast enhancement")

    # Few-shot parameters
    parser.add_argument("--n_shots", type=int, default=2,
                        help="Number of support examples")
    parser.add_argument("--proto_grid_size", type=int, default=8,
                        help="Grid size for prototype extraction")
    parser.add_argument("--align", action="store_true",
                        help="Use alignment loss")
    parser.add_argument("--val_wsize", type=int, default=1,
                        help="Window size for validation")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--n_steps", type=int, default=30000,
                        help="Maximum number of training steps")
    parser.add_argument("--optim_type", type=str, default="sgd",
                        choices=["sgd", "adam"], help="Optimizer type")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum (for SGD)")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Learning rate decay factor")

    # Other parameters
    parser.add_argument("--val_interval", type=int, default=1000,
                        help="Validation interval (iterations)")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Checkpoint saving interval (iterations)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID")

    config = parser.parse_args()
    train(config)
