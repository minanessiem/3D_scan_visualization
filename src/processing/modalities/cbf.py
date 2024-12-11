import torch
from monai.transforms import ScaleIntensityRange

def process_cbf(data: torch.Tensor, threshold_percent: float, max_val: float,
                ROI_mask: bool = False,
                description: str = None, **kwargs) -> torch.Tensor:
    """
    Process CBF images, returning either normalized data or thresholded mask based on ROI_mask flag
    
    Args:
        data: Input tensor
        threshold_percent: Percentile threshold for creating binary mask
        max_val: Maximum value for normalization
        ROI_mask: If True, return only threshold mask; if False, return normalized CBF values
        description: Optional description string
        **kwargs: Additional parameters (ignored)
    
    Returns:
        Tensor of shape [1, 1, H, W, D] containing either:
        - Normalized CBF values (if ROI_mask=False)
        - Binary mask where CBF < threshold_percent (if ROI_mask=True)
    """
    # Remove any extra dimensions to ensure data is [H, W, D]
    if data.ndim > 3:
        data = data.squeeze()
    
    # Clip values to physiologically relevant range
    data_clipped = torch.clamp(data, 0, 100)  # Retain values between 0 and max_val

    # Calculate the threshold value as a percentage of max_val
    cbf_threshold = (threshold_percent * max_val) / 100.0

    if ROI_mask:
        # Return only the thresholded mask (regions BELOW threshold)
        threshold_mask = (data < threshold_percent).float()
        return threshold_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W, D]
    else:
        # Return only the normalized data
        transform = ScaleIntensityRange(
            a_min=0,
            a_max=100,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data)
        return normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W, D]

def get_cbf_params(config: str, data_stats: dict) -> dict:
    """Get CBF-specific parameters for processing"""
    params = {
        'CBF_standard': {
            'threshold_percent': 30,
            'max_val': data_stats['max_val'],
            'ROI_mask': False,
            'description': 'Normalized CBF'
        },
        'CBF_3': {
            'threshold_percent': 3,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <3%'
        },
        'CBF_2': {
            'threshold_percent': 2,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <2%'
        },
        'CBF_30': {
            'threshold_percent': 30,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <30%'
        },
        'CBF_20': {
            'threshold_percent': 20,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <20%'
        },
        'CBF_15': {
            'threshold_percent': 15,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <15%'
        },
        'CBF_90': {
            'threshold_percent': 90,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'CBF <90%'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown CBF configuration: {config}")
        
    return params[config] 