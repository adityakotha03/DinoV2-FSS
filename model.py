import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProtoModule(nn.Module):
    """
    Prototype Module for the ALPNet model.
    """
    def __init__(self, proto_grid_size=8, feature_hw=[64, 64], embed_dim=256):
        super(ProtoModule, self).__init__()
        self.proto_grid_size = proto_grid_size
        self.feature_hw = feature_hw
        
        # Calculate kernel size for average pooling
        self.kernel_size = [ft_l // proto_grid_size for ft_l in feature_hw]
        self.avg_pool_op = nn.AvgPool2d(self.kernel_size)
        
    def safe_norm(self, x, p=2, dim=1, eps=1e-4):
        """Normalize features"""
        x_norm = torch.norm(x, p=p, dim=dim)
        x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
        x = x.div(x_norm.unsqueeze(1).expand_as(x))
        return x
    
    def get_prototypes(self, sup_x, sup_y, mode, thresh=0.95):
        """
        Extract prototypes from support image features - simplified version
        """
        if mode == 'mask':
            # Global prototype approach - average all features in the foreground mask
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + 1e-5)
            prototypes = proto
            proto_grid = sup_y.clone().detach()
            non_zero = torch.nonzero(proto_grid)
            
        elif mode == 'gridconv':
            # Simpler grid-based prototype approach
            
            # Use masked average pooling instead of complex indexing
            bs, c, h, w = sup_x.shape
            
            # Create empty list to store prototypes
            protos_list = []
            
            # Process each support image separately
            for i in range(bs):
                # Get features and mask for this support image
                feat_i = sup_x[i]  # [C, H, W]
                mask_i = sup_y[i]  # [1, H, W]
                
                # Apply threshold to the mask
                mask_i_bin = (mask_i > thresh).float()
                
                # Skip if no foreground
                if mask_i_bin.sum() < 1:
                    continue
                
                # Apply average pooling to both features and mask
                feat_pooled = self.avg_pool_op(feat_i.unsqueeze(0))[0]  # [C, h, w]
                mask_pooled = self.avg_pool_op(mask_i_bin)[0]  # [1, h, w]
                
                # Apply threshold to pooled mask
                mask_pooled_bin = (mask_pooled > 0.5).float()
                
                # Only keep grid cells with sufficient foreground
                valid_locations = mask_pooled_bin[0] > 0  # [h, w]
                
                # If no valid locations, skip this support image
                if valid_locations.sum() == 0:
                    continue
                
                # Extract features at valid locations
                for hi in range(valid_locations.shape[0]):
                    for wi in range(valid_locations.shape[1]):
                        if valid_locations[hi, wi]:
                            # Extract feature vector at this location
                            proto_feat = feat_pooled[:, hi, wi]  # [C]
                            protos_list.append(proto_feat)
            
            # Stack all prototypes
            if len(protos_list) > 0:
                protos = torch.stack(protos_list, dim=0)  # [num_protos, C]
                prototypes = self.safe_norm(protos)
            else:
                # Fallback to mask mode if no prototypes found
                print("Warning: No valid prototypes found in gridconv mode. Falling back to mask mode.")
                return self.get_prototypes(sup_x, sup_y, mode='mask', thresh=thresh)
                
            # Create prototype grid for visualization
            proto_grid = sup_y.clone().detach()
            proto_grid = (proto_grid > thresh).float()
            non_zero = torch.nonzero(proto_grid)
            
        return prototypes, proto_grid, non_zero
    
    def get_prediction(self, prototypes, query, mode):
        """
        Generate predictions using prototypes
        """
        if mode == 'mask':
            # Global prototype matching
            pred_mask = F.cosine_similarity(query, prototypes[..., None, None], dim=1, eps=1e-4) * 20.0
            pred_mask = pred_mask.max(dim=0)[0].unsqueeze(0)
            return pred_mask.unsqueeze(1), [pred_mask]
            
        elif mode == 'gridconv':
            # Local prototype matching with convolutional operation
            dists = F.conv2d(query, prototypes[..., None, None]) * 20
            pred_grid = torch.sum(F.softmax(dists, dim=1) * dists, dim=1, keepdim=True)
            debug_assign = dists.argmax(dim=1).float().detach()
            return pred_grid, [debug_assign]
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'mask' or 'gridconv'")
    
    def forward(self, qry, sup_x, sup_y, mode='gridconv', thresh=0.95):
        """
        Forward pass
        """
        # Normalize query features
        qry_n = qry if mode == 'mask' else self.safe_norm(qry)
        
        # Get prototypes from support images
        prototypes, proto_grid, proto_indices = self.get_prototypes(sup_x, sup_y, mode, thresh)
        
        # Generate predictions
        pred, debug_assign = self.get_prediction(prototypes, qry_n, mode)
        
        return pred, debug_assign, proto_grid

class FewShotSeg(nn.Module):
    """
    Few-shot segmentation model for eye fundus vessel segmentation
    """
    def __init__(self, image_size=512, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False, 'debug': False}
        
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
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.config['feature_hw'] = [self.image_size//14, self.image_size//14]
            
        elif self.config['which_model'] == 'dinov2_vitb':
            # Base ViT model from DINOv2
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.config['feature_hw'] = [self.image_size//14, self.image_size//14]
            
        elif self.config['which_model'] == 'dinov2_vitl':
            # Large ViT model from DINOv2
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.config['feature_hw'] = [self.image_size//14, self.image_size//14]
            
        else:
            # Default: use a ResNet backbone
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(pretrained=self.config.get('use_coco_init', True))
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
        
        self.cls_unit = ProtoModule(
            proto_grid_size=proto_hw,
            feature_hw=self.config["feature_hw"],
            embed_dim=embed_dim
        )
    
    def get_features(self, imgs):
        """Extract features from images"""
        if 'dinov2' in self.config.get('which_model', ''):
            # For DINOv2 models
            # Ensure image size is divisible by patch size
            patch_size = 14
            h, w = (self.image_size // patch_size) * patch_size, (self.image_size // patch_size) * patch_size
            imgs = F.interpolate(imgs, size=(h, w), mode='bilinear')
            
            # Extract features
            dino_fts = self.encoder.forward_features(imgs)
            img_fts = dino_fts["x_norm_patchtokens"]  # B, HW, C
            img_fts = img_fts.permute(0, 2, 1)  # B, C, HW
            
            # Reshape to spatial features
            C, HW = img_fts.shape[-2:]
            h = w = int(HW**0.5)
            img_fts = img_fts.view(-1, C, h, w)  # B, C, H, W
        else:
            # For ResNet backbone
            img_fts = self.encoder(imgs)['out']
            img_fts = self.feature_proj(img_fts)
        
        return img_fts
    
    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval=False, val_wsize=None):
        """
        Forward pass for few-shot segmentation
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        
        # Prepare support images
        support_images = torch.cat([torch.cat(way, dim=0) for way in supp_imgs], dim=0)
        
        # Concatenate with query images
        imgs_concat = torch.cat([support_images, torch.cat(qry_imgs, dim=0)], dim=0)
        
        # Extract features
        features = self.get_features(imgs_concat)
        
        # Split features into support and query (using contiguous for reshaping)
        supp_fts = features[:n_ways * n_shots].contiguous().view(n_ways, n_shots, -1, *features.shape[-2:])
        qry_fts = features[n_ways * n_shots:].contiguous().view(n_queries, -1, *features.shape[-2:])
        
        # Process support masks
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)
        back_mask = torch.stack([torch.stack(way, dim=0) for way in back_mask], dim=0)
        
        # Resize masks to match feature size
        fts_size = features.shape[-2:]
        fore_mask_resized = torch.stack([F.interpolate(fore_mask_w, size=fts_size, mode='nearest') 
                                        for fore_mask_w in fore_mask], dim=0)
        back_mask_resized = torch.stack([F.interpolate(back_mask_w, size=fts_size, mode='nearest') 
                                        for back_mask_w in back_mask], dim=0)
        
        # Compute segmentation scores
        outputs = []
        align_loss = 0
        
        # Background prediction
        bg_prototype_mode = 'gridconv'
        bg_score, bg_assign, bg_proto_grid = self.cls_unit(
            qry=qry_fts, 
            sup_x=supp_fts[:,0], 
            sup_y=back_mask_resized[:,0], 
            mode=bg_prototype_mode, 
            thresh=0.95
        )
        
        # Foreground prediction
        fg_prototype_mode = 'gridconv'
        fg_scores = []
        fg_assigns = []
        fg_proto_grids = []
        
        for way in range(n_ways):
            way_scores = []
            for shot in range(n_shots):
                fg_score, fg_assign, fg_proto_grid = self.cls_unit(
                    qry=qry_fts,
                    sup_x=supp_fts[way, shot].unsqueeze(0),
                    sup_y=fore_mask_resized[way, shot].unsqueeze(0),
                    mode=fg_prototype_mode,
                    thresh=0.95
                )
                way_scores.append(fg_score)
            
            # Combine scores from multiple shots (if applicable)
            combined_score = torch.stack(way_scores, dim=1).max(dim=1)[0]
            fg_scores.append(combined_score)
            fg_assigns.append(fg_assign)
            fg_proto_grids.append(fg_proto_grid)
        
        # Combine background and foreground scores
        pred = torch.cat([bg_score] + fg_scores, dim=1)  # N x (1 + ways) x H' x W'
        
        # Resize to original image size
        output = F.interpolate(pred, size=(self.image_size, self.image_size), mode='bilinear')
        
        # If in training mode, compute alignment loss
        if self.config['align'] and self.training:
            align_loss = self.compute_alignment_loss(
                qry_fts, pred, supp_fts, fore_mask_resized, back_mask_resized
            )
        
        return output, align_loss, [None, None], fg_assigns, fg_proto_grids, None, None
    
    def compute_alignment_loss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute alignment loss for training
        """
        n_ways = len(fore_mask)
        n_shots = len(fore_mask[0])
        
        # Get predicted segmentation
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  # 1 x N x H' x W'
        
        # Create binary masks for each class
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        
        # Compute loss for each way and shot
        loss = []
        for way in range(n_ways):
            for shot in range(n_shots):
                # Get query features and predicted mask
                qry_pred_fg_mask = F.interpolate(
                    binary_masks[way + 1].float(), 
                    size=qry_fts.shape[-2:], 
                    mode='bilinear'
                )
                
                qry_pred_bg_mask = F.interpolate(
                    binary_masks[0].float(),
                    size=qry_fts.shape[-2:],
                    mode='bilinear'
                )
                
                # Get support features
                img_fts = supp_fts[way:way+1, shot:shot+1]
                
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
                supp_pred = torch.cat([bg_score, fg_score], dim=1)
                
                # Resize to support mask size
                supp_pred = F.interpolate(
                    supp_pred,
                    size=fore_mask.shape[-2:],
                    mode='bilinear'
                )
                
                # Create support label
                supp_label = torch.full_like(
                    fore_mask[way, shot],
                    255,
                    device=img_fts.device
                ).long()
                
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                
                # Compute cross-entropy loss
                loss.append(
                    F.cross_entropy(
                        supp_pred.float(),
                        supp_label[None, ...],
                        ignore_index=255
                    ) / (n_shots * n_ways)
                )
        
        return torch.sum(torch.stack(loss))