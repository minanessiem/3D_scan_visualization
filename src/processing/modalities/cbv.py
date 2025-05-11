import torch
from monai.transforms import ScaleIntensityRange
"""
        'CBV_10ml': {
            'mask_threshold_min': None,
            'mask_threshold_max': None,
            'cbv_max_threshold': 10,
            'ROI_mask': False,
            'description': 'CBV <10 mL/100g'
        },
        'CBV_mask_core_10': {
            'mask_threshold_min': 0,
            'mask_threshold_max': 2,
            'cbv_max_threshold': 10,
            'ROI_mask': True,
            'description': 'CBV Mask Core <2 10 mL/100 g'
        },
        'CBV_mask_penumbra_10': {
            'mask_threshold_min': 2,
            'mask_threshold_max': 5,
            'cbv_max_threshold': 10,
            'ROI_mask': True,
            'description': 'CBV Mask <2 Penumbra <5 10 mL/100 g'
        },
        'CBV_mask_hyperemic_10': {
            'mask_threshold_min': 5,
            'mask_threshold_max': 10,
            'cbv_max_threshold': 10,
            'ROI_mask': True,
            'description': 'CBV Mask <5 Hyperemic <10 10 mL/100 g'
        },
        # ROI Mask configurations
        'CBV_mask_min_0_max_3': {
            'min_val': 0,
            'max_val': 3,
            'ROI_mask': True,
            'description': 'CBV Mask: 0 < CBV < 3.0 mL/100g'
        },
        'CBV_mask_min_0_max_3.5': {
            'min_val': 0,
            'max_val': 3.5,
            'ROI_mask': True,
            'description': 'CBV Mask: 0 < CBV < 3.5 mL/100g'
        },

"""
CBV_PARAMS = {
        # Normalization configurations
        'CBV_min_0_max_1.8': {
            'min_val': 0,
            'max_val': 1.8,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 1.8 mL/100g'
        },
        'CBV_min_0_max_1.9': {
            'min_val': 0,
            'max_val': 1.9,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 1.9 mL/100g'
        },
        'CBV_min_0_max_2': {
            'min_val': 0,
            'max_val': 2,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 2.0 mL/100g'
        },
        'CBV_min_0_max_2.2': {
            'min_val': 0,
            'max_val': 2.2,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 2.2 mL/100g'
        },
        'CBV_min_0_max_4': {
            'min_val': 0,
            'max_val': 4,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 4.0 mL/100g'
        },
        'CBV_min_0_max_6': {
            'min_val': 0,
            'max_val': 6,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 6.0 mL/100g'
        },
        'CBV_min_0_max_9': {
            'min_val': 0,
            'max_val': 9,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 9.0 mL/100g'
        },
        'CBV_min_0_max_maxval': {
            'min_val': 0,
            'max_val': 'MAX_VAL',
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < max value'
        },
    }


def process_cbv(data: torch.Tensor, min_val: float, max_val: float,
                description: str = None, **kwargs) -> torch.Tensor:
    """Process CBV images with min/max normalization"""
    data = torch.clamp(data, 0, max_val)

    transform = ScaleIntensityRange(
        a_min=min_val,
        a_max=max_val,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized[None, ...]


def get_cbv_params(config: str, data_stats: dict = None) -> dict:
    """Get CBV-specific parameters for processing"""
    # Deep copy to avoid modifying the original
    import copy
    
    if config == 'all':
        params = copy.deepcopy(CBV_PARAMS)
        
        # Replace placeholders with actual values if data_stats is provided
        if data_stats and 'min_val' in data_stats and 'max_val' in data_stats:
            for key in params:
                # Replace MAX_VAL placeholder
                if 'max_val' in params[key] and params[key]['max_val'] == 'MAX_VAL':
                    params[key]['max_val'] = float(data_stats['max_val'])
                
                # Replace MIN_VAL placeholder
                if 'min_val' in params[key] and params[key]['min_val'] == 'MIN_VAL':
                    params[key]['min_val'] = float(data_stats['min_val'])
        return params
    
    # Add CBV_ prefix if not present
    if not config.startswith('CBV_'):
        config = f'CBV_{config}'
        
    if config not in CBV_PARAMS:
        raise ValueError(f"Unknown CBV configuration: {config}")
    
    result = {config: copy.deepcopy(CBV_PARAMS[config])}
    
    # Replace placeholders with actual values if data_stats is provided
    if data_stats and 'min_val' in data_stats and 'max_val' in data_stats:
        # Replace MAX_VAL placeholder
        if 'max_val' in result[config] and result[config]['max_val'] == 'MAX_VAL':
            result[config]['max_val'] = float(data_stats['max_val'])
        
        # Replace MIN_VAL placeholder
        if 'min_val' in result[config] and result[config]['min_val'] == 'MIN_VAL':
            result[config]['min_val'] = float(data_stats['min_val'])
    
    return result