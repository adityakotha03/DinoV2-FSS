import numpy as np

def max_pooling_resize(mask, output_size):
    """
    Downsamples a 2D segmentation mask to 'output_size' using max pooling.
    
    Parameters:
        mask (np.ndarray): 2D array (H, W) representing the segmentation mask.
        output_size (tuple): Desired output size as (new_H, new_W).
    
    Returns:
        np.ndarray: The resized mask.
    """
    old_H, old_W = mask.shape
    new_H, new_W = output_size
    
    # Create an empty array for the pooled mask
    pooled_mask = np.zeros((new_H, new_W), dtype=mask.dtype)
    
    # Compute the ratio of old/new in each dimension
    row_scale = old_H / new_H
    col_scale = old_W / new_W
    
    for i in range(new_H):
        # Determine the vertical start and end (float-based) in the original image
        row_start = int(np.floor(i * row_scale))
        row_end   = int(np.floor((i + 1) * row_scale))
        
        # Make sure the last row covers up to the end
        if i == new_H - 1:
            row_end = old_H
        
        for j in range(new_W):
            # Determine the horizontal start and end (float-based) in the original image
            col_start = int(np.floor(j * col_scale))
            col_end   = int(np.floor((j + 1) * col_scale))
            
            # Make sure the last column covers up to the end
            if j == new_W - 1:
                col_end = old_W
            
            # Take the maximum over this window
            pooled_mask[i, j] = mask[row_start:row_end, col_start:col_end].max()
    
    return pooled_mask
