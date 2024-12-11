import torch
from monai.transforms import ScaleIntensityRange

def process_ncct(data: torch.Tensor, window: float, level: float, 
                description: str = None, **kwargs) -> torch.Tensor:
    """Process NCCT images with window/level normalization"""
    transform = ScaleIntensityRange(
        a_min=level - window/2,
        a_max=level + window/2,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized[None, ...]

def get_ncct_params(config: str) -> dict:
    """Get NCCT-specific parameters for processing"""
    params = {
        'NCCT_white': {
            'window': 8,
            'level': 27,
            'description': 'NCCT White matter window'
        },
        'NCCT_gray': {
            'window': 8,
            'level': 37,
            'description': 'NCCT Gray matter window'
        },
        'NCCT_wg': {
            'window': 49,
            'level': 35,
            'description': 'NCCT White and Gray matter window'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown NCCT configuration: {config}")
        
    return params[config] 