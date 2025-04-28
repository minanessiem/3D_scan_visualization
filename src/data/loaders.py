import os
import json
import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from ..processing.common import get_modality_params
from ..processing.modalities import (
    process_ncct, process_cta, process_cbf,
    process_cbv, process_mtt, process_tmax
)
from .transforms import preprocess_volume

def load_case_data(data_dir, json_file, case_id):
    # Load and parse JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Find the case in the JSON data
    case_data = None
    for item in data['training']:
        if item['caseID'] == case_id:
            case_data = item
            break
    
    if case_data is None:
        raise ValueError(f"Case ID {case_id} not found in the JSON file")
    
    return case_data

def load_and_process_nifti(file_path: str, modality_config: str, **kwargs) -> torch.Tensor:
    """
    Router function that loads NIfTI data and calls the appropriate processing function.
    
    Args:
        file_path: Path to the NIfTI file
        modality_config: Modality configuration (e.g., 'NCCT_white', 'CTA_arterial')
        **kwargs: Additional keyword arguments that can override default parameters
    
    Returns:
        Tensor of shape [1, C, H, W, D] where C varies by modality
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    original_spacing = img.header.get_zooms()[:3]
    
    print(f"\nDebug - Processing {modality_config}")
    print(f"NIfTI file: {os.path.basename(file_path)}")
    print(f"Original data shape: {data.shape}")
    print(f"Number of slices - Height: {data.shape[0]}, Width: {data.shape[1]}, Depth: {data.shape[2]}")
    print(f"Original spacing: {original_spacing}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    data_stats = {
        'min_val': float(np.min(data)),
        'max_val': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
    }
    
    base_modality, processing_params = get_modality_params(modality_config, data_stats)
    
    data = preprocess_volume(data, img.affine, original_spacing, mode="bilinear")
    
    processors = {
        'NCCT': process_ncct,
        'CTA': process_cta,
        'CBF': process_cbf,
        'CBV': process_cbv,
        'MTT': process_mtt,
        'TMAX': process_tmax,
    }
    
    if base_modality not in processors:
        raise ValueError(f"Unknown base modality: {base_modality}")
    
    processed_data = processors[base_modality](data, **processing_params)
    
    print(f"Processed {modality_config} using {processing_params.get('description', 'default')} configuration")
    print(f"Output shape: {processed_data.shape}")
    
    return processed_data 