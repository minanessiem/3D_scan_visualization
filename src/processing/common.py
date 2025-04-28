from typing import Tuple, Dict
from .modalities.ncct import get_ncct_params
from .modalities.cta import get_cta_params
from .modalities.cbf import get_cbf_params
from .modalities.cbv import get_cbv_params
from .modalities.mtt import get_mtt_params
from .modalities.tmax import get_tmax_params

def get_modality_params(modality_config: str, data_stats: dict = None) -> tuple:
    """
    Get processing parameters for a specific modality and configuration
    
    Args:
        modality_config: String identifier like 'NCCT_standard' or 'CTA_vessel'
        data_stats: Optional dictionary with data statistics (min_val, max_val, etc.)
        
    Returns:
        Tuple of (base_modality, params_dict) where base_modality is one of 
        'NCCT', 'CTA', etc. and params_dict contains processing parameters
    """
    # Split base modality from specific configuration
    parts = modality_config.split('_', 1)
    base_modality = parts[0]
    
    # Default to full config name if no underscore is present
    config = modality_config if len(parts) == 1 else modality_config
    
    # Get parameters based on modality type
    if base_modality == 'NCCT':
        params = get_ncct_params(config, data_stats)
    elif base_modality == 'CTA':
        params = get_cta_params(config, data_stats)
    elif base_modality == 'CBF':
        params = get_cbf_params(config, data_stats)
    elif base_modality == 'CBV':
        params = get_cbv_params(config, data_stats)
    elif base_modality == 'MTT':
        params = get_mtt_params(config, data_stats)
    elif base_modality == 'TMAX':
        params = get_tmax_params(config, data_stats)
    else:
        raise ValueError(f"Unknown modality: {base_modality}")
    
    # Extract the first (and typically only) value from the returned dictionary
    if params:
        # Get the first key (usually the config_name itself)
        first_key = next(iter(params))
        return base_modality, params[first_key]
    
    raise ValueError(f"No parameters found for configuration: {config}") 