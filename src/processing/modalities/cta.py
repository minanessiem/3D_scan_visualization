import torch
from monai.transforms import ScaleIntensityRange

def process_cta(data: torch.Tensor, window: float, level: float,
                clip_range: tuple, description: str = None, **kwargs) -> torch.Tensor:
    """Process CTA images with window/level normalization"""
    transform = ScaleIntensityRange(
        a_min=level - window/2,
        a_max=level + window/2,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized[None, ...]

def get_cta_params(config: str) -> dict:
    """Get CTA-specific parameters for processing"""
    params = {
        'CTA_arterial': {
            'window': 600,
            'level': 150,
            'clip_range': (-1000, 1000),
            'description': 'Arterial phase'
        },
        'CTA_venous': {
            'window': 400,
            'level': 100,
            'clip_range': (-1000, 1000),
            'description': 'Venous phase'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown CTA configuration: {config}")
        
    return params[config] 