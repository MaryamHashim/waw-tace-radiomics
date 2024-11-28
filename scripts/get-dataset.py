import json
import os
import SimpleITK as sitk
from lib import DATASET_PATH
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


def get_spacing(path: str) -> tuple[float, float, float]:
    if "total_segmentator" in path:
        return None
    mm_filter = sitk.MinimumMaximumImageFilter()
    image = sitk.ReadImage(path)
    mm_filter.Execute(image)
    return {
        "path": path,
        "spacing": image.GetSpacing(),
        "minimum": mm_filter.GetMinimum(),
        "maximum": mm_filter.GetMaximum(),
    }


n_workers = 8
path = DATASET_PATH
mask_path = f"{DATASET_PATH}/tumor_masks_wawtace_v1_08_05_2024"
path = Path(path)
mask_path = Path(mask_path)

masks = {}
for mask in mask_path.rglob("*nrrd"):
    mask = str(mask)
    identifier = mask.split(os.sep)[-2]
    if identifier not in masks:
        masks[identifier] = []
    masks[identifier].append(mask)

all_data = {}
with Pool(n_workers) as p:
    for idx in [0, 1, 2, 3]:
        all_scans = [str(x) for x in path.rglob(f"*_{idx}_*nii.gz")]
        spacing_iterator = p.imap_unordered(get_spacing, all_scans)
        for spacing in tqdm(spacing_iterator, total=len(all_scans)):
            if spacing is None:
                continue
            else:
                identifier = spacing["path"].split(os.sep)[-2]
                if identifier not in all_data:
                    all_data[identifier] = {}
                all_data[identifier]["mask"] = (
                    masks[identifier] if identifier in masks else None
                )
                all_data[identifier][idx] = spacing

with open("dataset.json", "w") as o:
    json.dump(all_data, o)
