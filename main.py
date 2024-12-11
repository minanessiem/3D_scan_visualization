import argparse
import os
import nibabel as nib
import numpy as np
from src.data.loaders import load_case_data, load_and_process_nifti
from src.data.transforms import preprocess_volume
from src.utils.validation import slice_range, modality_list, get_max_slices
from src.visualization.plotting import plot_multimodal_slices

def main():
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

if __name__ == "__main__":
    main() 