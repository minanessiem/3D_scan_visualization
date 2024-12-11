import torch
from monai.transforms import ScaleIntensityRange

def process_cbv(data: torch.Tensor, threshold_value: float = 2.0, max_val: float = 8.0,
                ROI_mask: bool = False, description: str = None, **kwargs) -> torch.Tensor:
    """
    Process CBV images for ischemic core detection or normalization.

    Args:
        data (torch.Tensor): Input tensor representing CBV values.
        threshold_value (float): Threshold for ischemic core (default: 2.0 mL/100 g).
        max_val (float): Maximum value for normalization (default: 8.0 mL/100 g).
        ROI_mask (bool): If True, return binary mask for ischemic core; otherwise, normalize data.
        description (str): Optional description for logging.

    Returns:
        torch.Tensor: Normalized CBV values or binary mask.
    """
    # Ensure input tensor has appropriate shape
    if data.ndim > 3:
        data = data.squeeze()
    
    # Clip values to a reasonable range for CBV (e.g., [0, max_val])
    data_clipped = torch.clamp(data, 0, max_val)
    
    if ROI_mask:
        # Generate binary mask for ischemic core (CBV < threshold_value)
        threshold_mask = (data_clipped < threshold_value).float()
        return threshold_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    else:
        # Normalize CBV for visualization
        transform = ScaleIntensityRange(
            a_min=0,
            a_max=max_val,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data_clipped)
        return normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def get_cbv_params(config: str) -> dict:
    """Get CBV-specific parameters for processing"""
    params = {
        'CBV_standard': {
            'threshold_value': 2.0,
            'max_val': 8.0,
            'ROI_mask': False,
            'description': 'Normalized CBV values'
        },
        'CBV_core': {
            'threshold_value': 2.0,
            'max_val': 8.0,
            'ROI_mask': True,
            'description': 'Ischemic core (CBV < 2.0 mL/100 g)'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown CBV configuration: {config}")
        
    return params[config] 