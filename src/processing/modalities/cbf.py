import torch
from monai.transforms import ScaleIntensityRange

# Define parameters without referencing data_stats
CBF_PARAMS = {
    # Physiological range configurations
    'CBF_min_minval_max_maxval': {
        'threshold_percent': None,
        'min_val': 'MIN_VAL',
        'max_val': 'MAX_VAL',
        'ROI_mask': False,
        'description': 'minval < CBF < maxval mL / 100 g / min (full range)'
    },
    'CBF_min_0_max_250mL': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 250,
        'ROI_mask': False,
        'description': '0 < CBF < 250 mL / 100 g / min (physiological)'
    },
    
    # Cell death threshold configurations (medically relevant ranges)
    'CBF_min_0_max_10': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 10,
        'ROI_mask': False,
        'description': '0 < CBF < 10 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_10': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 10,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 10 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_12': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 12,
        'ROI_mask': False,
        'description': '0 < CBF < 12 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_12': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 12,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 12 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_15': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 15,
        'ROI_mask': False,
        'description': '0 < CBF < 15 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_15': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 15,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 15 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_18': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 18,
        'ROI_mask': False,
        'description': '0 < CBF < 18 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_18': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 18,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 18 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_20': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 20,
        'ROI_mask': False,
        'description': '0 < CBF < 20 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_20': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 20,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 20 mL / 100 g / min (cell death threshold)'
    },
    # Identification of penumbra and infarct in acute ischemic stroke using computed tomography perfusion-derived blood flow and blood volume measurements, Murphy et al., 2006
    'CBF_min_0_max_25': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 25,
        'ROI_mask': False,
        'description': '0 < CBF < 25 mL / 100 g / min'
    },
    'CBF_min_0_max_30': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 30,
        'ROI_mask': False,
        'description': '0 < CBF < 30 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_30': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 30,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 30 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_40': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 40,
        'ROI_mask': False,
        'description': '0 < CBF < 40 mL / 100 g / min (cell death threshold)'
    },
    'CBF_mask_min_0_max_40': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 40,
        'ROI_mask': True,
        'description': 'Mask: 0 < CBF < 40 mL / 100 g / min (cell death threshold)'
    },
    'CBF_min_0_max_50': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 50,
        'ROI_mask': False,
        'description': '0 < CBF < 50 mL / 100 g / min'
    },
    'CBF_min_0_max_60': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 60,
        'ROI_mask': False,
        'description': '0 < CBF < 60 mL / 100 g / min'
    },
    'CBF_min_0_max_70': {
        'threshold_percent': None,
        'min_val': 0,
        'max_val': 70,
        'ROI_mask': False,
        'description': '0 < CBF < 70 mL / 100 g / min'
    },

}

def process_cbf(data: torch.Tensor, threshold_percent: float, min_val: float, max_val: float,
                ROI_mask: bool = False,
                description: str = None, **kwargs) -> torch.Tensor:
    """
    Process CBF images, returning either normalized data or thresholded mask based on ROI_mask flag
    
    Args:
        data: Input tensor
        threshold_percent: Percentile threshold for creating binary mask or setting upper normalization limit
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization or threshold calculation
        ROI_mask: If True, return only threshold mask; if False, return normalized CBF values
        description: Optional description string
        **kwargs: Additional parameters (ignored)
    
    Returns:
        Tensor of shape [1, H, W, D] containing either:
        - Normalized CBF values (if ROI_mask=False)
        - Binary mask where CBF < threshold_percent (if ROI_mask=True)
    """
    
    # Step 1: Create a more robust brain mask (exclude background and noise)
    brain_mask = (data > 1e-5).float()  # Exclude very low or zero values (background)

    # Clip values to physiologically relevant range
    data_clipped = torch.clamp(data, 0, max_val)  # Retain values between 0 and max_val

    # Calculate threshold value if threshold_percent is provided
    cbf_threshold = (threshold_percent * max_val) / 100.0 if threshold_percent is not None else max_val

    if ROI_mask:
        # Return only the thresholded mask (regions BELOW threshold)
        threshold_mask = ((data_clipped < cbf_threshold) & (brain_mask > min_val)).float()
        return threshold_mask[None, ...]  # Add channel dimension
    else:
        # Return normalized data using either threshold or max_val as the upper bound
        transform = ScaleIntensityRange(
            a_min=min_val,
            a_max=cbf_threshold,  # Use threshold as upper bound if threshold_percent is provided
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data)
        return normalized[None, ...]  # Add channel dimension

def get_cbf_params(config: str, data_stats: dict = None) -> dict:
    """Get CBF-specific parameters for processing"""
    if config == 'all':
        # Deep copy to avoid modifying the original
        import copy
        params = copy.deepcopy(CBF_PARAMS)
        
        # Replace placeholders with actual values if data_stats is provided
        if data_stats:
            for key in params:
                # Replace MAX_VAL placeholder
                if 'max_val' in data_stats and params[key].get('max_val') == 'MAX_VAL':
                    params[key]['max_val'] = data_stats['max_val']
                
                # Replace MIN_VAL placeholder
                if 'min_val' in data_stats and params[key].get('min_val') == 'MIN_VAL':
                    params[key]['min_val'] = data_stats['min_val']
        return params
    
    # Add CBF_ prefix if not present
    if not config.startswith('CBF_'):
        config = f'CBF_{config}'
        
    if config not in CBF_PARAMS:
        raise ValueError(f"Unknown CBF configuration: {config}")
    
    # Deep copy to avoid modifying the original
    import copy
    result = {config: copy.deepcopy(CBF_PARAMS[config])}
    
    # Replace placeholders with actual values if data_stats is provided
    if data_stats:
        # Replace MAX_VAL placeholder
        if 'max_val' in data_stats and result[config].get('max_val') == 'MAX_VAL':
            result[config]['max_val'] = data_stats['max_val']
        
        # Replace MIN_VAL placeholder
        if 'min_val' in data_stats and result[config].get('min_val') == 'MIN_VAL':
            result[config]['min_val'] = data_stats['min_val']
        
    return result 