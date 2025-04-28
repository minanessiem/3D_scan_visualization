import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import torch
from .grid import (
    calculate_grid_dimensions,
    calculate_composite_dimensions,
    create_empty_composite,
    add_modality_to_composite,
    add_segmentation_overlay
)

def get_slice_from_volume(volume, slice_num, orientation='axial'):
    """
    Extract a 2D slice from a 3D volume with proper orientation.
    
    Args:
        volume: Input volume of shape [1, C, H, W, D] or [C, H, W, D]
        slice_num: Slice number to extract
        orientation: One of 'axial', 'coronal', 'sagittal'
    
    Returns:
        2D slice as numpy array
    """
    # Handle numpy arrays
    if isinstance(volume, np.ndarray):
        if volume.ndim == 4:  # [1, H, W, D]
            volume = volume[0]  # Now [H, W, D]
        if orientation == 'axial':
            return volume[:, :, slice_num].T
        elif orientation == 'coronal':
            return volume[:, slice_num, :].T
        elif orientation == 'sagittal':
            return volume[slice_num, :, :].T
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
    
    # Handle torch tensors
    # Remove batch dimension if present
    if volume.ndim == 5:
        volume = volume.squeeze(0)
    
    # Get the first channel if multi-channel
    if volume.shape[0] > 1:
        volume = volume[0]  # Take first channel for visualization
    else:
        volume = volume.squeeze(0)  # Remove channel dim if single channel
    
    # Now volume shape is [H, W, D]
    if orientation == 'axial':
        slice_data = volume[:, :, slice_num].T
    elif orientation == 'coronal':
        slice_data = volume[:, slice_num, :].T
    elif orientation == 'sagittal':
        slice_data = volume[slice_num, :, :].T
    else:
        raise ValueError(f"Invalid orientation: {orientation}")
    
    # Convert to numpy
    if isinstance(slice_data, torch.Tensor):
        slice_data = slice_data.cpu().numpy()
    
    return slice_data

def create_subplot(ax, composite, slice_num):
    """Create and configure a single subplot"""
    im = ax.imshow(composite.astype(np.uint8), origin='lower')
    ax.axis('off')
    ax.text(0.02, 0.98, f'Slice {slice_num}', 
            transform=ax.transAxes, color='black',
            fontsize=10, va='top')
    return ax

def setup_figure(num_slices, grid_dims):
    """Create and configure the matplotlib figure"""
    fig = plt.figure(figsize=(20, 20))
    return fig

def plot_multimodal_slices(data_dict: Dict[str, torch.Tensor], 
                          segmentation_data: Optional[np.ndarray] = None, 
                          slice_nums: Optional[list] = None, 
                          alpha: float = 0.3, 
                          orientation: str = 'axial'):
    """Plot multiple modalities with composite images for each slice"""
    # Calculate grid dimensions
    mod_grid_rows, mod_grid_cols = calculate_grid_dimensions(len(data_dict))
    print(len(slice_nums), slice_nums)
    # Get slice information
    sample_slice = get_slice_from_volume(next(iter(data_dict.values())), 
                                       slice_nums[0], orientation)
    slice_height, slice_width = sample_slice.shape
    
    # Calculate composite dimensions
    composite_height, composite_width = calculate_composite_dimensions(
        slice_height, slice_width, mod_grid_rows, mod_grid_cols
    )
    
    # Process each slice
    composite_slices = []
    for slice_num in slice_nums:
        composite = create_empty_composite(composite_height, composite_width)
        
        # Add each modality
        for mod_idx, (_, data) in enumerate(data_dict.items()):
            # Calculate position from top-left instead of bottom-left
            row = mod_idx // mod_grid_cols
            col = mod_idx % mod_grid_cols
            # Adjust y_start to start from top
            y_start = (mod_grid_rows - 1 - row) * slice_height
            x_start = col * slice_width
            position = (y_start, x_start)
            dimensions = (slice_height, slice_width)
            
            slice_data = get_slice_from_volume(data, slice_num, orientation)
            composite = add_modality_to_composite(
                composite, slice_data, position, dimensions
            )
            
            if segmentation_data is not None:
                seg_slice = get_slice_from_volume(
                    segmentation_data, slice_num, orientation
                )
                composite = add_segmentation_overlay(
                    composite, seg_slice, position, dimensions, alpha
                )
        
        composite_slices.append(composite)
    
    # Create visualization
    slice_grid_rows, slice_grid_cols = calculate_grid_dimensions(len(slice_nums))
    fig = setup_figure(len(slice_nums), (slice_grid_rows, slice_grid_cols))
    
    # Create subplots
    for idx, composite in enumerate(composite_slices):
        ax = plt.subplot(slice_grid_rows, slice_grid_cols, idx + 1)
        create_subplot(ax, composite, slice_nums[idx])
    
    plt.suptitle(f'{orientation.capitalize()} Slices - Multiple Modalities', 
                fontsize=16)
    plt.tight_layout()
    plt.show() 