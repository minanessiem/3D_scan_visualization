import torch
from monai.transforms import ScaleIntensityRange

def process_mtt(data: torch.Tensor, window: float = 12, level: float = 6,
                threshold_min: float = 4, threshold_max: float = 14,
                ROI_mask: bool = False, description: str = None, **kwargs) -> torch.Tensor:
    """
    Process MTT images by applying window/level normalization or generating a binary ROI mask
    for values within a given threshold range.

    Args:
        data (torch.Tensor): Input tensor representing MTT values.
        window (float): Window size for normalization.
        level (float): Level for normalization.
        threshold_min (float): Minimum threshold for MTT ROI mask.
        threshold_max (float): Maximum threshold for MTT ROI mask.
        ROI_mask (bool): If True, return binary ROI mask; otherwise, normalized MTT map.
        description (str): Optional description string.

    Returns:
        torch.Tensor: Normalized MTT values or binary ROI mask.
    """
    if data.ndim > 3:
        data = data.squeeze()

    # Step 1: Clip negative values to 0 (non-physiological MTT)
    data_clipped = torch.clamp(data, min=0)

    # Step 2: Create a brain mask (exclude background values close to zero)
    brain_mask = (data_clipped > 1e-5).float()

    if ROI_mask:
        # Step 3a: Generate binary mask for values within [threshold_min, threshold_max]
        roi_mask = ((data_clipped >= threshold_min) & (data_clipped <= threshold_max)).float()
        roi_mask = roi_mask * brain_mask  # Exclude background
        return roi_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    else:
        # Step 3b: Apply window/level normalization for MTT visualization
        a_min = level - window / 2  # Lower window boundary
        a_max = level + window / 2  # Upper window boundary
        transform = ScaleIntensityRange(
            a_min=a_min,
            a_max=a_max,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data_clipped) * brain_mask  # Normalize and exclude background
        return normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions


def get_mtt_params(config: str) -> dict:
    """Get MTT-specific parameters for processing"""
    params = {
        'MTT_standard': {
            'window': 12,
            'level': 6,
            'description': 'Standard MTT processing'
        },
        'MTT_4_14': {
            'window': 12,
            'level': 6,
            'threshold_min': 4,
            'threshold_max': 14,
            'ROI_mask': True,
            'description': 'MTT ROI Mask (4–14s)'
        },
    }
    
    if config not in params:
        raise ValueError(f"Unknown MTT configuration: {config}")
        
    return params[config] 