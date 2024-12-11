import torch
from monai.transforms import ScaleIntensityRange

def process_mtt(data: torch.Tensor, window: float, level: float,
                description: str = None, **kwargs) -> torch.Tensor:
    """Process MTT images with window/level normalization"""
    transform = ScaleIntensityRange(
        a_min=level - window/2,
        a_max=level + window/2,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized[None, ...]

def get_mtt_params(config: str) -> dict:
    """Get MTT-specific parameters for processing"""
    params = {
        'MTT_standard': {
            'window': 12,
            'level': 6,
            'description': 'Standard MTT processing'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown MTT configuration: {config}")
        
    return params[config] 