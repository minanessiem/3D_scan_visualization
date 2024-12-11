import torch
from monai.transforms import ScaleIntensityRange

def process_tmax(data: torch.Tensor, threshold: float, max_val: float,
                ROI_mask: bool = False,
                description: str = None, **kwargs) -> torch.Tensor:
    """
    Process TMAX images, returning either normalized data or thresholded mask based on ROI_mask flag
    
    Args:
        data: Input tensor
        threshold: Threshold value for creating binary mask
        max_val: Maximum value for normalization
        ROI_mask: If True, return only threshold mask; if False, return normalized TMAX values
        description: Optional description string
        **kwargs: Additional parameters (ignored)
    
    Returns:
        Tensor of shape [1, 1, H, W, D] containing either:
        - Normalized TMAX values (if ROI_mask=False)
        - Binary mask where TMAX > threshold (if ROI_mask=True)
    """
    # Remove any extra dimensions to ensure data is [H, W, D]
    if data.ndim > 3:
        data = data.squeeze()
    
    if ROI_mask:
        # Return only the thresholded mask
        threshold_mask = (data > threshold).float()
        return threshold_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W, D]
    else:
        # Return only the normalized data
        transform = ScaleIntensityRange(
            a_min=0,
            a_max=max_val,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data)
        return normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W, D]

def get_tmax_params(config: str, data_stats: dict) -> dict:
    """Get TMAX-specific parameters for processing"""
    params = {
        'TMAX_standard': {
            'threshold': 3,
            'max_val': data_stats['max_val'],
            'ROI_mask': False,
            'description': 'TMAX > 3'
        },
        'TMAX_3': {
            'threshold': 3,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX > 3'
        },
        'TMAX_6': {
            'threshold': 6,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX > 6'
        },
        'TMAX_8': {
            'threshold': 8,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX > 6'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown TMAX configuration: {config}")
        
    return params[config] 