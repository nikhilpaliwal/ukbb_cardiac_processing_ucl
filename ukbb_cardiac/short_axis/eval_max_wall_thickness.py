# Copyright 2025, Nikhil Paliwal. All Rights Reserved.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt


def compute_wall_thickness(seg):
    """
    Compute maximum myocardial wall thickness from a 3D segmentation volume.

    :param seg: 3D numpy array with segmentation labels:
                0 = background, 1 = LV cavity, 2 = myocardium, 3 = RV cavity
    :return: Maximum wall thickness in mm
    """
    spacing = seg.header.get_zooms()  # voxel spacing (x, y, z) in mm
    seg_data = seg.get_fdata().astype(np.uint8)

    mask_myo = (seg_data == 2)

    # Compute distance transform within myocardium
    dt = distance_transform_edt(mask_myo, sampling=spacing)
    max_thickness = dt.max() * 2  # thickness = radius * 2
    return max_thickness


def main(args):
    all_cases = sorted(os.listdir(args.data_dir))
    results = []

    for case_id in all_cases:
        case_dir = os.path.join(args.data_dir, case_id)
        sa_seg_file = os.path.join(case_dir, 'seg_sa.nii.gz')

        if not os.path.exists(sa_seg_file):
            print(f"[Warning] Segmentation not found: {sa_seg_file}")
            continue

        try:
            seg = nib.load(sa_seg_file)
            max_thickness = compute_wall_thickness(seg)
            results.append({'CaseID': case_id, 'MaxWallThickness_mm': max_thickness})
            print(f"{case_id}: Max wall thickness = {max_thickness:.2f} mm")
        except Exception as e:
            print(f"[Error] Failed for {case_id}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved maximum wall thickness results to: {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate max myocardial wall thickness from short-axis segmentations.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing subject folders with seg_sa.nii.gz files.')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output CSV file for maximum wall thickness.')

    args = parser.parse_args()
    main(args)
