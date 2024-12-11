import sys
import os
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from monai.transforms import ScaleIntensityRange, Spacing, Orientation, Affine
from monai.data import MetaTensor
import torch
from src.utils.validation import slice_range, modality_list, get_max_slices

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

def apply_isotropic_transform(data: MetaTensor, mode: str = "bilinear") -> torch.Tensor:
    """
    Resample a 3D volume stored in a MetaTensor to isotropic spacing (1.0, 1.0, 1.0)
    using the MONAI Orientation and Spacing transforms.
    """

    if "pixdim" not in data.meta:
        raise ValueError("No 'pixdim' found in metadata. Ensure the input is a MetaTensor with spatial metadata.")

    original_spacing = data.meta["pixdim"]
    if len(original_spacing) < 3:
        raise ValueError("pixdim metadata does not contain at least 3 spatial dimensions.")

    # If data is (B, C, H, W, D), squeeze out the batch dimension
    # Make sure you only do this if B=1
    if data.ndim == 5 and data.shape[0] == 1:
        data = data.squeeze(0)  # now shape: (C, H, W, D)

    # Reorient the image
    orientation_transform = Orientation(axcodes="RAS")
    data = orientation_transform(data)

    # Define target isotropic spacing
    target_spacing = (1.0, 1.0, 1.0)

    # Apply Spacing
    spacing_transform = Spacing(
        pixdim=target_spacing,
        mode=mode,
        padding_mode="border",
        dtype=torch.float32
    )
    resampled = spacing_transform(data)

    return resampled.as_tensor()

def preprocess_volume(img_data: np.ndarray, affine: np.ndarray, spacing: tuple, mode: str = "bilinear") -> torch.Tensor:
    """
    Preprocess a volume by converting it to a MetaTensor and applying orientation and spacing transforms.
    
    Args:
        img_data: Input volume data as numpy array
        affine: Affine matrix from the NIfTI header
        spacing: Original voxel spacing (pixdim)
        mode: Interpolation mode ("bilinear" for images, "nearest" for segmentations)
    
    Returns:
        Preprocessed volume as a torch.Tensor
    """
    # Add batch (N=1) and channel (C=1) dimensions
    data_tensor = torch.as_tensor(img_data)[None, None]
    
    # Create MetaTensor with metadata
    meta_tensor = MetaTensor(
        data_tensor,
        affine=torch.as_tensor(affine, dtype=torch.float32),
        meta={"pixdim": spacing}
    )
    
    # Apply isotropic transform
    processed_data = apply_isotropic_transform(meta_tensor, mode=mode)
    
    return processed_data

def get_modality_params(modality_config: str, data_stats: dict) -> tuple[str, dict]:
    """
    Get modality-specific parameters for processing based on configuration.
    
    Args:
        modality_config: The modality configuration (e.g., 'NCCT_white', 'NCCT_gray')
        data_stats: Dictionary containing basic data statistics
    
    Returns:
        Tuple of (base_modality, parameters)
    """
    params = {
        # NCCT configurations
        'NCCT_white': {
            'window': 8,
            'level': 27,
            # 'clip_range': (-1000, 1000),
            'description': 'NCCT White matter window'
        },
        'NCCT_gray': {
            'window': 8,
            'level': 37,
            # 'clip_range': (-1000, 1000),
            'description': 'NCCT Gray matter window'
        },
        'NCCT_wg': {
            'window': 49,
            'level': 35,
            # 'clip_range': (-1000, 1000),
            'description': 'NCCT White and Gray matter window'
        },
        
        # CTA configurations
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
        },
        
        # CBF configurations
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
        },
        
        # CBV configurations
        'CBV_standard': {
            'threshold_value': 2.0,
            'max_val': 8.0,
            'ROI_mask': False,
            'description': 'Normalized CBV values'
        },
        # Binary mask for ischemic core
        'CBV_core': {
            'threshold_value': 2.0,
            'max_val': 8.0,
            'ROI_mask': True,
            'description': 'Ischemic core (CBV < 2.0 mL/100 g)'
        },
        
        # MTT configurations
        'MTT_standard': {
            'window': 12,
            'level': 6,
            'description': 'Standard MTT processing'
        },
        
        'TMAX_standard': {
            'threshold': 3,
            'max_val': data_stats['max_val'],
            'ROI_mask': False,
            'description': 'TMAX > 3'
        },
        # TMAX configurations
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
    
    if modality_config not in params:
        raise ValueError(f"Modality configuration not found: {modality_config}")
    
    base_modality = modality_config.split('_')[0]
    return base_modality, params[modality_config]

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
    print(f"Original spacing: {original_spacing}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    data_stats = {
        'min_val': float(np.min(data)),
        'max_val': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
    }
    
    base_modality, config_params = get_modality_params(modality_config, data_stats)
    processing_params = {**config_params, **kwargs}
    
    data = preprocess_volume(data, img.affine, original_spacing, mode="bilinear")
    
    processors = {
        'NCCT': process_ncct,
        'CTA': process_cta,
        'CBF': process_cbf,
        'CBV': process_cbv,
        'MTT': process_mtt,
        'TMAX': process_tmax
    }
    
    if base_modality not in processors:
        raise ValueError(f"Unknown base modality: {base_modality}")
    
    processed_data = processors[base_modality](data, **processing_params)
    print(f"Processed {modality_config} using {processing_params['description']} configuration")
    print(f"Output shape: {processed_data.shape}")
    
    return processed_data

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

def process_cbv(data: torch.Tensor, threshold_value: float = 2.0, max_val: float = 8.0,
                ROI_mask: bool = False, description: str = None, **kwargs) -> torch.Tensor:
    """
    Process CBV images for ischemic core detection or normalization.

    Args:
        data (torch.Tensor): Input tensor representing CBV values.
        threshold_value (float): Threshold for ischemic core (default: 2.0 mL/100 g).
        max_val (float): Maximum value for normalization (default: 8.0 mL/100 g).
        ROI_mask (bool): If True, return binary mask for ischemic core; otherwise, normalize data.
        description (str): Optional description for logging.

    Returns:
        torch.Tensor: Normalized CBV values or binary mask.
    """
    # Ensure input tensor has appropriate shape
    if data.ndim > 3:
        data = data.squeeze()
    
    # Clip values to a reasonable range for CBV (e.g., [0, max_val])
    data_clipped = torch.clamp(data, 0, max_val)
    
    if ROI_mask:
        # Generate binary mask for ischemic core (CBV < threshold_value)
        threshold_mask = (data_clipped < threshold_value).float()
        return threshold_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    else:
        # Normalize CBV for visualization
        transform = ScaleIntensityRange(
            a_min=0,
            a_max=max_val,
            b_min=0,
            b_max=1,
            clip=True
        )
        normalized = transform(data_clipped)
        return normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions


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

def slice_range(arg):
    """Custom argparse type for handling slice ranges"""
    try:
        # Handle single number
        if arg.isdigit():
            return [int(arg)]
        
        # Handle range 'min,max'
        start, end = map(int, arg.split(','))
        return list(range(start, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Slice must be a single number or 'min,max' range"
        )

def modality_list(arg):
    """Custom argparse type for handling comma-separated modalities"""
    return [mod.strip() for mod in arg.split(',')]

def get_slice_from_volume(volume, slice_num, orientation='axial'):
    """
    Extract a 2D slice from a 3D volume with proper orientation.
    
    Args:
        volume: Input volume of shape [1, C, H, W, D] or [C, H, W, D]
        slice_num: Slice number to extract
        orientation: One of 'axial', 'coronal', 'sagittal'
    
    Returns:
        2D slice as numpy array
    """
    # Handle numpy arrays
    if isinstance(volume, np.ndarray):
        if volume.ndim == 4:  # [1, H, W, D]
            volume = volume[0]  # Now [H, W, D]
        if orientation == 'axial':
            return volume[:, :, slice_num].T
        elif orientation == 'coronal':
            return volume[:, slice_num, :].T
        elif orientation == 'sagittal':
            return volume[slice_num, :, :].T
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
    
    # Handle torch tensors
    # Remove batch dimension if present
    if volume.ndim == 5:
        volume = volume.squeeze(0)
    
    # Get the first channel if multi-channel
    if volume.shape[0] > 1:
        volume = volume[0]  # Take first channel for visualization
    else:
        volume = volume.squeeze(0)  # Remove channel dim if single channel
    
    # Now volume shape is [H, W, D]
    if orientation == 'axial':
        slice_data = volume[:, :, slice_num].T
    elif orientation == 'coronal':
        slice_data = volume[:, slice_num, :].T
    elif orientation == 'sagittal':
        slice_data = volume[slice_num, :, :].T
    else:
        raise ValueError(f"Invalid orientation: {orientation}")
    
    # Convert to numpy
    if isinstance(slice_data, torch.Tensor):
        slice_data = slice_data.cpu().numpy()
    
    return slice_data

def get_max_slices(volume, orientation):
    """Get the maximum number of slices for a given orientation"""
    # Handle numpy arrays
    if isinstance(volume, np.ndarray):
        if volume.ndim == 4:  # [1, H, W, D]
            volume = volume[0]  # Now [H, W, D]
        if orientation == 'axial':
            return volume.shape[2]
        elif orientation == 'coronal':
            return volume.shape[1]
        elif orientation == 'sagittal':
            return volume.shape[0]
        else:
            raise ValueError(f"Invalid orientation: {orientation}")
    
    # Handle torch tensors
    # Remove batch dimension if present
    if volume.ndim == 5:
        volume = volume.squeeze(0)
    
    # Remove channel dimension
    if volume.ndim == 4:
        volume = volume.squeeze(0)
    
    # Now volume shape is [H, W, D]
    if orientation == 'axial':
        return volume.shape[2]
    elif orientation == 'coronal':
        return volume.shape[1]
    elif orientation == 'sagittal':
        return volume.shape[0]
    else:
        raise ValueError(f"Invalid orientation: {orientation}")

def calculate_grid_dimensions(num_items):
    """Calculate optimal grid dimensions for a square-ish layout"""
    rows = int(np.ceil(np.sqrt(num_items)))
    cols = rows  # Force square grid for modalities
    return rows, cols

def calculate_composite_dimensions(slice_height, slice_width, grid_rows, grid_cols):
    """Calculate dimensions for composite image"""
    return slice_height * grid_rows, slice_width * grid_cols

def create_empty_composite(height, width):
    """Create white background canvas for composite image"""
    return np.ones((height, width, 3)) * 255

def add_modality_to_composite(composite, slice_data, position, dimensions):
    """Add a single modality slice to the composite image
    
    Args:
        composite: The composite image array
        slice_data: The 2D slice data to add
        position: Tuple of (y_start, x_start)
        dimensions: Tuple of (height, width)
    """
    y_start, x_start = position
    height, width = dimensions
    
    # Normalize slice data to [0, 255]
    normalized = ((slice_data - slice_data.min()) / 
                 (slice_data.max() - slice_data.min()) * 255)
    
    # Add to all RGB channels
    composite[y_start:y_start + height, 
             x_start:x_start + width, :] = normalized[..., None]
    
    return composite

def add_segmentation_overlay(composite, seg_data, position, dimensions, alpha=0.3):
    """Add segmentation overlay to the composite image"""
    if seg_data is None:
        return composite
        
    y_start, x_start = position
    height, width = dimensions
    
    seg_overlay = np.zeros((height, width, 4))
    seg_overlay[seg_data > 0] = [1, 0, 0, alpha]
    
    mask = seg_overlay[..., 3:4]
    composite[y_start:y_start + height, 
             x_start:x_start + width, :3] = (
        composite[y_start:y_start + height, 
                 x_start:x_start + width, :3] * (1 - mask) +
        seg_overlay[..., :3] * 255 * mask
    )
    
    return composite

def create_subplot(ax, composite, slice_num):
    """Create and configure a single subplot"""
    im = ax.imshow(composite.astype(np.uint8), origin='lower')
    ax.axis('off')
    ax.text(0.02, 0.98, f'Slice {slice_num}', 
            transform=ax.transAxes, color='black',
            fontsize=10, va='top')
    return ax

def setup_figure(num_slices, grid_dims):
    """Create and configure the matplotlib figure"""
    fig = plt.figure(figsize=(20, 20))
    return fig

def plot_multimodal_slices(data_dict, segmentation_data=None, slice_nums=None, 
                          alpha=0.3, orientation='axial'):
    """Plot multiple modalities with composite images for each slice"""
    # Calculate grid dimensions
    mod_grid_rows, mod_grid_cols = calculate_grid_dimensions(len(data_dict))
    
    # Get slice information
    sample_slice = get_slice_from_volume(next(iter(data_dict.values())), 
                                       slice_nums[0], orientation)
    slice_height, slice_width = sample_slice.shape
    
    # Calculate composite dimensions
    composite_height, composite_width = calculate_composite_dimensions(
        slice_height, slice_width, mod_grid_rows, mod_grid_cols
    )
    
    # Process each slice
    composite_slices = []
    for slice_num in slice_nums:
        composite = create_empty_composite(composite_height, composite_width)
        
        # Add each modality
        for mod_idx, (_, data) in enumerate(data_dict.items()):
            # Calculate position from top-left instead of bottom-left
            row = mod_idx // mod_grid_cols
            col = mod_idx % mod_grid_cols
            # Adjust y_start to start from top
            y_start = (mod_grid_rows - 1 - row) * slice_height  # Changed this line
            x_start = col * slice_width
            position = (y_start, x_start)
            dimensions = (slice_height, slice_width)
            
            slice_data = get_slice_from_volume(data, slice_num, orientation)
            composite = add_modality_to_composite(
                composite, slice_data, position, dimensions
            )
            
            if segmentation_data is not None:
                seg_slice = get_slice_from_volume(
                    segmentation_data, slice_num, orientation
                )
                composite = add_segmentation_overlay(
                    composite, seg_slice, position, dimensions, alpha
                )
        
        composite_slices.append(composite)
    
    # Create visualization
    slice_grid_rows, slice_grid_cols = calculate_grid_dimensions(len(slice_nums))
    fig = setup_figure(len(slice_nums), (slice_grid_rows, slice_grid_cols))
    
    # Create subplots
    for idx, composite in enumerate(composite_slices):
        ax = plt.subplot(slice_grid_rows, slice_grid_cols, idx + 1)
        create_subplot(ax, composite, slice_nums[idx])
    
    plt.suptitle(f'{orientation.capitalize()} Slices - Multiple Modalities', 
                fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize medical imaging data')
    parser.add_argument("--data_dir", default=r"..\\isles24_combined\\", type=str, help="dataset directory")
    parser.add_argument("--json_list", default=r"..\\isles24_combined\\isles24_multimodal_5fold_allmasks_full.json",
                        type=str, help="dataset json file")
    parser.add_argument("--case_id", required=True, type=str, help="case ID to visualize")
    parser.add_argument("--modalities", 
                       default="NCCT_white,NCCT_gray", 
                       type=modality_list,
                       help="comma-separated list of modality configurations to visualize")
    parser.add_argument("--slices", type=slice_range, 
                       help="slice selection (single number or 'min,max' range)")
    parser.add_argument("--show_segmentation_mask", action="store_true", 
                       help="overlay segmentation mask if available")
    parser.add_argument("--orientation", default="axial", choices=['axial', 'sagittal', 'coronal'],
                       help="orientation of slices to display")

    args = parser.parse_args()

    # Load case data from JSON
    case_data = load_case_data(args.data_dir, args.json_list, args.case_id)
    
    # Load segmentation mask if requested
    segmentation_data = None
    if args.show_segmentation_mask and 'label' in case_data:
        try:
            seg_path = os.path.join(args.data_dir, case_data['label'])
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata().astype(np.int32)
            
            print(f"\nProcessing segmentation mask from: {seg_path}")
            print(f"Original segmentation shape: {seg_data.shape}")
            
            # Preprocess the segmentation mask using nearest neighbor interpolation
            seg_iso = preprocess_volume(
                seg_data, 
                seg_img.affine, 
                seg_img.header.get_zooms()[:3],
                mode="nearest"
            )
            
            print(f"Isotropic segmentation shape: {seg_iso.shape}")
            segmentation_data = seg_iso.round().int().numpy()
            
        except Exception as e:
            print(f"Error loading segmentation mask: {str(e)}")
            segmentation_data = None
    
    # Process each requested modality configuration
    processed_data = {}
    for modality_config in args.modalities:
        # Extract base modality name (everything before the underscore)
        base_modality = modality_config.split('_')[0]
        
        if base_modality not in case_data:
            print(f"Warning: Base modality {base_modality} not found for case {args.case_id}")
            continue
            
        # Handle NCCT which is stored as a list
        if isinstance(case_data[base_modality], list):
            # Take the first NCCT file
            file_path = os.path.join(args.data_dir, case_data[base_modality][0])
        else:
            file_path = os.path.join(args.data_dir, case_data[base_modality])
            
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            processed = load_and_process_nifti(file_path, modality_config)
            processed_data[modality_config] = processed
        except Exception as e:
            print(f"Error processing {modality_config} for case {args.case_id}: {str(e)}")
    
    # Parse slice selection if provided
    slice_nums = None
    if args.slices and processed_data:
        num_slices = get_max_slices(next(iter(processed_data.values())), args.orientation)
        # Validate the range
        if any(s < 0 or s >= num_slices for s in args.slices):
            raise ValueError(f"Slice numbers must be between 0 and {num_slices-1}")
        slice_nums = args.slices
    
    # Plot all processed modalities together
    if processed_data:
        plot_multimodal_slices(processed_data, segmentation_data, slice_nums, orientation=args.orientation)