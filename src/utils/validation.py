import argparse
import numpy as np
import torch

def slice_range(arg):
    """Custom argparse type for handling slice ranges"""
    try:
        # Handle single number
        if arg.isdigit():
            return [int(arg)]
        
        # Handle range 'min,max'
        start, end = map(int, arg.split(','))
        return list(range(start, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Slice must be a single number or 'min,max' range"
        )

def modality_list(arg):
    """Custom argparse type for handling comma-separated modalities"""
    return [mod.strip() for mod in arg.split(',')]

def get_max_slices(volume, orientation):
    """Get the maximum number of slices for a given orientation"""
    # Handle numpy arrays
    if isinstance(volume, np.ndarray):
        if volume.ndim == 4:  # [1, H, W, D]
            volume = volume[0]  # Now [H, W, D]
        if orientation == 'axial':
            return volume.shape[2]
        elif orientation == 'coronal':
            return volume.shape[1]
        elif orientation == 'sagittal':
            return volume.shape[0]
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
    
    # Handle torch tensors
    # Remove batch dimension if present
    if volume.ndim == 5:
        volume = volume.squeeze(0)
    
    # Remove channel dimension
    if volume.ndim == 4:
        volume = volume.squeeze(0)
    
    # Now volume shape is [H, W, D]
    if orientation == 'axial':
        return volume.shape[2]
    elif orientation == 'coronal':
        return volume.shape[1]
    elif orientation == 'sagittal':
        return volume.shape[0]
    else:
        raise ValueError(f"Invalid orientation: {orientation}") 