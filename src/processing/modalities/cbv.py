import torch
from monai.transforms import ScaleIntensityRange

def process_cbv(data: torch.Tensor, threshold_min: float = 0.5, threshold_max: float = 3.5,
                core_threshold: float = 2.0, ROI_mask: bool = False, description: str = None) -> torch.Tensor:
    """
    Process CBV images to generate an ROI mask or normalized CBV values.

    Args:
        data (torch.Tensor): Input tensor representing CBV values.
        threshold_min (float): Minimum threshold for ROI range (default: 0.5 mL/100 g).
        threshold_max (float): Maximum threshold for ROI range (default: 3.5 mL/100 g).
        core_threshold (float): Threshold for ischemic core detection (default: 2.0 mL/100 g).
        ROI_mask (bool): If True, return binary mask; otherwise, return normalized CBV values.
        description (str): Optional description for logging.

    Returns:
        torch.Tensor: Binary ROI mask or normalized CBV map.
    """
    if data.ndim > 3:
        data = data.squeeze()

    # Step 1: Remove invalid (negative) CBV values
    data_clipped = torch.clamp(data, min=0)

    # Step 2: Create a brain mask (exclude background where values == 0)
    brain_mask = (data_clipped > 1e-5).float()

    if ROI_mask:
        # Step 3: Generate binary ROI mask for CBV in [threshold_min, threshold_max]
        roi_mask = ((data_clipped >= threshold_min) & (data_clipped <= threshold_max)).float()
        final_mask = roi_mask * brain_mask  # Exclude background
        return final_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    else:
        # Step 4: Normalize CBV to [0, 1] within the given range [threshold_min, threshold_max]
        transform = ScaleIntensityRange(
            a_min=threshold_min,
            a_max=threshold_max,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data_clipped) * brain_mask  # Exclude background
        return normalized.unsqueeze(0).unsqueeze(0)

def get_cbv_params(config: str) -> dict:
    """Get CBV-specific parameters for processing"""
    params = {
        'CBV_core': {
            'threshold_min': 0.5,
            'threshold_max': 3.5,
            'core_threshold': 2.0,
            'ROI_mask': True,
            'description': 'CBV Core Mask (CBV < 2.0 mL/100g)'
        },

        'CBV_standard': {
            'threshold_min': 0.5,
            'threshold_max': 3.5,
            'ROI_mask': False,
            'description': 'Normalized CBV Map (0.5–3.5 mL/100g)'
        }

    }
    
    if config not in params:
        raise ValueError(f"Unknown CBV configuration: {config}")
        
    return params[config] 