import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_norm(x, p=2, dim=1, eps=1e-4):
    """Normalize features"""
    x_norm = torch.norm(x, p=p, dim=dim)
    x_norm = torch.max(x_norm, torch.ones_like(x_norm).to(x.device) * eps)
    x = x.div(x_norm.unsqueeze(1).expand_as(x))
    return x


class ProtoModule(nn.Module):
    """
    Prototype Module for the ALPNet model.
    Computes prototype-based segmentation.
    """

    def __init__(self, proto_grid_size=8, feature_hw=[32, 32], embed_dim=256):
        """
        Args:
            proto_grid_size: Grid size for multi-prototyping
            feature_hw: Spatial size of input feature maps
            embed_dim: Embedding dimension
        """
        super(ProtoModule, self).__init__()
        self.proto_grid_size = proto_grid_size
        self.feature_hw = feature_hw

        # Calculate kernel size for average pooling
        self.kernel_size = [ft_l // grid_l for ft_l,
                            grid_l in zip(feature_hw, [proto_grid_size, proto_grid_size])]
        self.avg_pool_op = nn.AvgPool2d(self.kernel_size)

    def get_prototypes(self, sup_x, sup_y, mode, thresh=0.95):
        """
        Extract prototypes from support image features

        Args:
            sup_x: Support image features [B, C, H, W]
            sup_y: Support image masks [B, 1, H, W]
            mode: Prototype extraction mode ('mask' or 'gridconv')
            thresh: Threshold for considering a grid cell as foreground

        Returns:
            prototypes: Normalized prototypes
            proto_grid: Grid showing prototype assignment
            non_zero: Indices of non-zero elements in proto_grid
        """
        if mode == 'mask':
            # Global prototype (average features in the mask)
            # Sum features where mask is 1, then divide by number of mask pixels
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            prototypes = proto
            proto_grid = sup_y.clone().detach()
            non_zero = torch.nonzero(proto_grid)

        elif mode == 'gridconv':
            # Local prototypes (grid-based)
            nch = sup_x.shape[1]  # Number of channels
            batch_size = sup_x.shape[0]  # Batch size

            # Apply average pooling to get grid-level features
            n_sup_x = self.avg_pool_op(sup_x)  # [B, C, grid_h, grid_w]
            
            # Reshape to prepare for extraction
            n_sup_x = n_sup_x.view(batch_size, nch, -1)  # [B, C, grid_h*grid_w]
            n_sup_x = n_sup_x.permute(0, 2, 1)  # [B, grid_h*grid_w, C]
            n_sup_x = n_sup_x.reshape(-1, nch)  # [B*grid_h*grid_w, C]

            # Apply average pooling to masks
            sup_y_g = self.avg_pool_op(sup_y)  # [B, 1, grid_h, grid_w]

            # Create prototype grid
            proto_grid = sup_y_g.clone().detach()
            proto_grid[proto_grid < thresh] = 0
            non_zero = torch.nonzero(proto_grid)

            # Flatten sup_y_g to use as a selector
            sup_y_g = sup_y_g.view(batch_size, 1, -1)  # [B, 1, grid_h*grid_w]
            sup_y_g = sup_y_g.reshape(-1)  # [B*grid_h*grid_w]

            # Get features for grid cells above threshold
            protos = n_sup_x[sup_y_g > thresh, :]  # [num_prototypes, C]

            if protos.shape[0] == 0:
                print("Warning: Failed to find prototypes, falling back to mask mode")
                return self.get_prototypes(sup_x, sup_y, mode='mask', thresh=thresh)

            # Normalize prototypes
            prototypes = safe_norm(protos)

        return prototypes, proto_grid, non_zero

    def get_prediction(self, prototypes, query, mode):
        """
        Generate predictions using prototypes

        Args:
            prototypes: Extracted prototypes [num_prototypes, C]
            query: Query image features [B, C, H, W]
            mode: Prediction mode ('mask' or 'gridconv')

        Returns:
            pred: Prediction map
            debug_assign: Debug information for visualization
        """
        if mode == 'mask':
            # Global prototype matching
            # Expand prototypes to [num_prototypes, C, 1, 1] for broadcasting
            expanded_prototypes = prototypes.unsqueeze(-1).unsqueeze(-1)
            
            # Compute cosine similarity between query features and prototypes
            pred_mask = F.cosine_similarity(
                query, expanded_prototypes, dim=1, eps=1e-4
            ) * 20.0  # Scale factor
            
            # Get maximum similarity across prototypes
            pred_mask = pred_mask.max(dim=0)[0].unsqueeze(0).unsqueeze(1)
            return pred_mask, [pred_mask]

        elif mode == 'gridconv':
            # Local prototype matching with convolutional operation
            # Use prototypes as convolutional kernels
            prototypes_kernel = prototypes.unsqueeze(-1).unsqueeze(-1)  # [num_prototypes, C, 1, 1]
            
            # Compute similarities using convolution
            dists = F.conv2d(query, prototypes_kernel) * 20  # Scaled similarities
            
            # Weighted sum of similarities (soft-assignment)
            pred_grid = torch.sum(
                F.softmax(dists, dim=1) * dists, 
                dim=1, 
                keepdim=True
            )
            
            # For visualization: hard assignment of prototypes
            debug_assign = dists.argmax(dim=1).float().detach()
            
            return pred_grid, [debug_assign]

        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'mask' or 'gridconv'")

    def forward(self, qry, sup_x, sup_y, mode='gridconv', thresh=0.95):
        """
        Forward pass

        Args:
            qry: Query image features [B, C, H, W]
            sup_x: Support image features [B, C, H, W]
            sup_y: Support image masks [B, 1, H, W]
            mode: Operation mode ('mask' or 'gridconv')
            thresh: Threshold for prototype extraction

        Returns:
            pred: Prediction map
            debug_assign: Debug information
            proto_grid: Grid showing prototype locations
        """
        # Normalize query features if in gridconv mode
        qry_n = qry if mode == 'mask' else safe_norm(qry)

        # Get prototypes from support images
        prototypes, proto_grid, proto_indices = self.get_prototypes(
            sup_x, sup_y, mode, thresh)

        # Generate predictions
        pred, debug_assign = self.get_prediction(prototypes, qry_n, mode)

        return pred, debug_assign, proto_grid


class FewShotSeg(nn.Module):
    """
    Few-shot segmentation model for eye segmentation
    """

    def __init__(self, image_size=224, pretrained_path=None, cfg=None):
        """
        Args:
            image_size: Input image size
            pretrained_path: Path to pretrained weights
            cfg: Model configuration
        """
        super(FewShotSeg, self).__init__()
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False, 'debug': False, 'which_model': 'resnet50'}

        # Get encoder
        self.get_encoder()

        # Get prototype module
        self.get_cls()

        # Load pretrained weights if provided
        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path), strict=True)
            print(f'Pre-trained model {self.pretrained_path} has been loaded')

    def get_encoder(self):
        """Set up the encoder (feature extractor)"""
        self.config['feature_hw'] = [self.image_size//16, self.image_size//16]

        if self.config['which_model'] == 'dinov2_vits':
            # Small ViT model from DINOv2
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14')
            self.config['feature_hw'] = [
                self.image_size//14, self.image_size//14]

        elif self.config['which_model'] == 'dinov2_vitb':
            # Base ViT model from DINOv2
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14')
            self.config['feature_hw'] = [
                self.image_size//14, self.image_size//14]

        elif self.config['which_model'] == 'dinov2_vitl':
            # Large ViT model from DINOv2
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitl14')
            self.config['feature_hw'] = [
                self.image_size//14, self.image_size//14]

        else:
            # Default: use a ResNet backbone
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(
                pretrained=self.config.get('use_coco_init', True))
            self.encoder = model.backbone
            self.feature_proj = nn.Conv2d(2048, 256, kernel_size=1)

    def get_cls(self):
        """Set up the prototype classifier"""
        proto_hw = self.config.get("proto_grid_size", 8)

        # Determine embedding dimension based on model
        embed_dim = 384  # Default for ViT-S
        if 'vitb' in self.config.get('which_model', ''):
            embed_dim = 768
        elif 'vitl' in self.config.get('which_model', ''):
            embed_dim = 1024
        elif 'resnet' in self.config.get('which_model', ''):
            embed_dim = 256

        self.cls_unit = ProtoModule(
            proto_grid_size=proto_hw,
            feature_hw=self.config["feature_hw"],
            embed_dim=embed_dim
        )

    def get_features(self, imgs):
        """Extract features from images"""
        if 'dinov2' in self.config.get('which_model', ''):
            # For DINOv2 models
            patch_size = 14
            # Explicitly calculate the resized dimensions divisible by patch_size
            h, w = imgs.shape[2], imgs.shape[3]
            new_h, new_w = (h // patch_size) * \
                patch_size, (w // patch_size) * patch_size
            imgs = F.interpolate(imgs, size=(new_h, new_w),
                                 mode='bilinear', align_corners=False)

            # Extract features
            dino_fts = self.encoder.forward_features(imgs)
            img_fts = dino_fts["x_norm_patchtokens"]  # B, HW, C
            img_fts = img_fts.permute(0, 2, 1)  # B, C, HW

            # Explicit spatial dimensions
            h_feat = new_h // patch_size
            w_feat = new_w // patch_size
            img_fts = img_fts.view(-1, img_fts.size(1),
                                   h_feat, w_feat)  # B, C, H', W'
        else:
            # For ResNet backbone
            img_fts = self.encoder(imgs)['out']
            img_fts = self.feature_proj(img_fts)

        return img_fts

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval=False, val_wsize=None):
        """
        Forward pass for few-shot segmentation

        Args:
            supp_imgs: Support images [batch_size, n_shot, channel, h, w]
            fore_mask: Foreground masks for support images [batch_size, n_shot, 1, h, w]
            back_mask: Background masks for support images [batch_size, n_shot, 1, h, w]
            qry_imgs: Query images [batch_size, 1, channel, h, w]
            isval: Whether in validation mode
            val_wsize: Window size for validation

        Returns:
            output: Segmentation output
            align_loss: Alignment loss value (for training)
        """
        # Get dimensions
        B, n_shot, C, H, W = supp_imgs.shape
        n_ways = 1  # Assuming 1-way few-shot segmentation (foreground vs background)
        
        # Reshape support and query images
        supp_imgs_flat = supp_imgs.view(B * n_shot, C, H, W)
        qry_imgs_flat = qry_imgs.view(B, C, H, W)
        
        # Extract features
        supp_feat = self.get_features(supp_imgs_flat)
        qry_feat = self.get_features(qry_imgs_flat)
        
        # Get feature sizes
        _, _, h_feat, w_feat = supp_feat.shape
        
        # Reshape support features back to [B, n_shot, C, h, w]
        supp_feat = supp_feat.view(B, n_shot, -1, h_feat, w_feat)
        print(f"Support images features shape: {supp_feat.shape}")
        
        # Resize masks to match feature size
        fore_mask_flat = fore_mask.view(B * n_shot, 1, H, W)
        back_mask_flat = back_mask.view(B * n_shot, 1, H, W)
        
        fore_mask_resized = F.interpolate(fore_mask_flat, size=(h_feat, w_feat), mode='nearest')
        back_mask_resized = F.interpolate(back_mask_flat, size=(h_feat, w_feat), mode='nearest')
        
        # Reshape resized masks back to [B, n_shot, 1, h, w]
        fore_mask_resized = fore_mask_resized.view(B, n_shot, 1, h_feat, w_feat)
        back_mask_resized = back_mask_resized.view(B, n_shot, 1, h_feat, w_feat)
        
        print(f"Support fore ground mask features shape: {fore_mask_resized.shape}")
        
        # Initialize variables for storing outputs
        align_loss = 0
        fg_scores = []
        fg_assigns = []
        fg_proto_grids = []
        
        # Background prediction using all support images
        bg_prototype_mode = 'gridconv'
        
        # For background, use all support images
        bg_scores = []
        for shot in range(n_shot):
            bg_score, bg_assign, bg_proto_grid = self.cls_unit(
                qry=qry_feat,  # Use all query features
                sup_x=supp_feat[:, shot],  # Use features from this shot
                sup_y=back_mask_resized[:, shot],  # Use background mask
                mode=bg_prototype_mode,
                thresh=0.95
            )
            bg_scores.append(bg_score)
        
        # Combine background scores from multiple shots (max pooling)
        bg_score = torch.stack(bg_scores, dim=1).max(dim=1)[0] if n_shot > 1 else bg_scores[0]
        
        # Foreground prediction
        fg_prototype_mode = 'gridconv'
        
        # For each way (typically just 1 in binary segmentation)
        for way in range(n_ways):
            way_scores = []
            for shot in range(n_shot):
                fg_score, fg_assign, fg_proto_grid = self.cls_unit(
                    qry=qry_feat,
                    sup_x=supp_feat[:, shot],
                    sup_y=fore_mask_resized[:, shot],
                    mode=fg_prototype_mode,
                    thresh=0.95
                )
                way_scores.append(fg_score)
            
            # Combine scores from multiple shots (if applicable)
            if n_shot > 1:
                combined_score = torch.stack(way_scores, dim=1).max(dim=1)[0]
            else:
                combined_score = way_scores[0]
                
            fg_scores.append(combined_score)
            fg_assigns.append(fg_assign)
            fg_proto_grids.append(fg_proto_grid)
        
        # Combine background and foreground scores
        # Shape: B x (1 + n_ways) x H' x W'
        pred = torch.cat([bg_score] + fg_scores, dim=1)
        
        # Resize to original image size
        output = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
        
        # If in training mode and alignment is configured, compute alignment loss
        if self.config.get('align', False) and self.training:
            align_loss = self.compute_alignment_loss(
                qry_feat, pred, supp_feat, fore_mask_resized, back_mask_resized
            )
        
        return output, align_loss, [None, None], fg_assigns, fg_proto_grids, None, None

    def compute_alignment_loss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute alignment loss for training

        This loss encourages consistent prototype assignments between
        support and query images
        
        Args:
            qry_fts: Query image features [B, C, H, W]
            pred: Predicted segmentation [B, 2, H, W] (background, foreground)
            supp_fts: Support image features [B, n_shot, C, H, W]
            fore_mask: Foreground masks [B, n_shot, 1, H, W]
            back_mask: Background masks [B, n_shot, 1, H, W]
            
        Returns:
            loss: Alignment loss value
        """
        B, n_shot = fore_mask.shape[0], fore_mask.shape[1]
        n_ways = 1  # Binary segmentation (foreground vs background)

        # Get predicted segmentation
        pred_mask = pred.argmax(dim=1, keepdim=True)  # [B, 1, H, W]

        # Create binary masks for each class (background=0, foreground=1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # [background_mask, foreground_mask]

        # Compute loss for each way and shot
        losses = []
        
        for way in range(n_ways):
            for shot in range(n_shot):
                # Get binary masks from prediction on query image
                qry_pred_fg_mask = F.interpolate(
                    binary_masks[way + 1].float(),
                    size=qry_fts.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                qry_pred_bg_mask = F.interpolate(
                    binary_masks[0].float(),
                    size=qry_fts.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                # Get support features for current shot
                img_fts = supp_fts[:, shot]  # [B, C, H, W]

                # Predict support image using query prototypes
                bg_score, _, _ = self.cls_unit(
                    qry=img_fts,
                    sup_x=qry_fts,
                    sup_y=qry_pred_bg_mask,
                    mode='gridconv',
                    thresh=0.95
                )

                fg_score, _, _ = self.cls_unit(
                    qry=img_fts,
                    sup_x=qry_fts,
                    sup_y=qry_pred_fg_mask,
                    mode='gridconv',
                    thresh=0.95
                )

                # Combine scores
                supp_pred = torch.cat([bg_score, fg_score], dim=1)  # [B, 2, H, W]

                # Resize to support mask size
                supp_pred = F.interpolate(
                    supp_pred,
                    size=fore_mask.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                # Create support label 
                # 0 for background, 1 for foreground, 255 for ignore
                supp_label = torch.full_like(
                    fore_mask[:, shot, 0],  # [B, H, W]
                    255,
                    device=img_fts.device
                ).long()

                # Set foreground and background labels
                supp_label[fore_mask[:, shot, 0] == 1] = 1
                supp_label[back_mask[:, shot, 0] == 1] = 0

                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    supp_pred,
                    supp_label,
                    ignore_index=255,
                    reduction='mean'
                ) / (n_shot * n_ways)
                
                losses.append(loss)

        return torch.sum(torch.stack(losses))