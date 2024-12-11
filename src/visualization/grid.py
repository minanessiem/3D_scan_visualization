import numpy as np

def calculate_grid_dimensions(num_items):
    """Calculate optimal grid dimensions for a square-ish layout"""
    rows = int(np.ceil(np.sqrt(num_items)))
    cols = rows  # Force square grid for modalities
    return rows, cols

def calculate_composite_dimensions(slice_height, slice_width, grid_rows, grid_cols):
    """Calculate dimensions for composite image"""
    return slice_height * grid_rows, slice_width * grid_cols

def create_empty_composite(height, width):
    """Create white background canvas for composite image"""
    return np.ones((height, width, 3)) * 255

def add_modality_to_composite(composite, slice_data, position, dimensions):
    """Add a single modality slice to the composite image
    
    Args:
        composite: The composite image array
        slice_data: The 2D slice data to add
        position: Tuple of (y_start, x_start)
        dimensions: Tuple of (height, width)
    """
    y_start, x_start = position
    height, width = dimensions
    
    # Normalize slice data to [0, 255]
    normalized = ((slice_data - slice_data.min()) / 
                 (slice_data.max() - slice_data.min()) * 255)
    
    # Add to all RGB channels
    composite[y_start:y_start + height, 
             x_start:x_start + width, :] = normalized[..., None]
    
    return composite

def add_segmentation_overlay(composite, seg_data, position, dimensions, alpha=0.3):
    """Add segmentation overlay to the composite image"""
    if seg_data is None:
        return composite
        
    y_start, x_start = position
    height, width = dimensions
    
    seg_overlay = np.zeros((height, width, 4))
    seg_overlay[seg_data > 0] = [1, 0, 0, alpha]
    
    mask = seg_overlay[..., 3:4]
    composite[y_start:y_start + height, 
             x_start:x_start + width, :3] = (
        composite[y_start:y_start + height, 
                 x_start:x_start + width, :3] * (1 - mask) +
        seg_overlay[..., :3] * 255 * mask
    )
    
    return composite 