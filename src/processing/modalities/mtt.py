import torch
from monai.transforms import ScaleIntensityRange
import copy

MTT_PARAMS = {
    'MTT_min_0_max_4': {
        'min_val': 0,
        'max_val': 4,
        'description': 'MTT normalized: 0 < MTT < 4'
    },
    'MTT_min_0_max_6': {
        'min_val': 0,
        'max_val': 6,
        'description': 'MTT normalized: 0 < MTT < 6'
    },
    'MTT_min_0_max_8': {
        'min_val': 0,
        'max_val': 8,
        'description': 'MTT normalized: 0 < MTT < 8'
    },
    'MTT_min_0_max_10': {
        'min_val': 0,
        'max_val': 10,
        'description': 'MTT normalized: 0 < MTT < 10'
    },
    'MTT_min_0_max_12': {
        'min_val': 0,
        'max_val': 12,
        'description': 'MTT normalized: 0 < MTT < 12'
    },
    'MTT_min_0_max_16': {
        'min_val': 0,
        'max_val': 16,
        'description': 'MTT normalized: 0 < MTT < 16'
    },
    'MTT_min_0_max_30': {
        'min_val': 0,
        'max_val': 30,
        'description': 'MTT normalized: 0 < MTT < 30'
    },
    'MTT_min_0_max_maxval': {
        'min_val': 0,
        'max_val': 'MAX_VAL',
        'description': 'MTT with direct min to max scaling (full range)'
    },
    'MTT_min_4_max_10': {
        'min_val': 4,
        'max_val': 10,
        'description': 'MTT normalized: 4 < MTT < 10'
    },
    'MTT_min_6_max_10': {
        'min_val': 6,
        'max_val': 10,
        'description': 'MTT normalized: 6 < MTT < 10'
    },
    'MTT_min_4_max_30': {
        'min_val': 4,
        'max_val': 30,
        'description': 'MTT normalized: 4 < MTT < 30'
    },
    'MTT_min_6_max_30': {
        'min_val': 6,
        'max_val': 30,
        'description': 'MTT normalized: 6 < MTT < 30'
    },
    'MTT_min_8_max_30': {
        'min_val': 8,
        'max_val': 30,
        'description': 'MTT normalized: 8 < MTT < 30'
    },
    'MTT_min_10_max_30': {
        'min_val': 10,
        'max_val': 30,
        'description': 'MTT normalized: 10 < MTT < 30'
    },
}

def process_mtt(data: torch.Tensor, min_val: float, max_val: float,
                description: str = None, **kwargs) -> torch.Tensor:
    """Process MTT images with window/level normalization"""
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

def get_mtt_params(config: str, data_stats: dict = None) -> dict:
    """Get MTT-specific parameters for processing"""
    # Deep copy to avoid modifying the original
    
    if config == 'all':
        params = copy.deepcopy(MTT_PARAMS)
        
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
    
    # Add MTT_ prefix if not present
    if not config.startswith('MTT_'):
        config = f'MTT_{config}'
        
    if config not in MTT_PARAMS:
        raise ValueError(f"Unknown MTT configuration: {config}")
    
    result = {config: copy.deepcopy(MTT_PARAMS[config])}
    
    # Replace placeholders with actual values if data_stats is provided
    if data_stats and 'min_val' in data_stats and 'max_val' in data_stats:
        # Replace MAX_VAL placeholder
        if 'max_val' in result[config] and result[config]['max_val'] == 'MAX_VAL':
            result[config]['max_val'] = float(data_stats['max_val'])
        
        # Replace MIN_VAL placeholder
        if 'min_val' in result[config] and result[config]['min_val'] == 'MIN_VAL':
            result[config]['min_val'] = float(data_stats['min_val'])
    
    return result 