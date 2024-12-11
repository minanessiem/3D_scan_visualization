import numpy as np
import torch
from monai.transforms import Orientation, Spacing
from monai.data import MetaTensor
from typing import Tuple

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