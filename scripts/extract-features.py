import os
import json
import numpy as np
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor
from radiomics import setVerbosity
from tqdm import tqdm
from lib import FEATURE_DIR

setVerbosity(60)


class FeatureExtractors:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.feature_extractors = {
            "1": RadiomicsFeatureExtractor(),
            "2": RadiomicsFeatureExtractor(),
        }

        for k in self.feature_extractors:
            self.feature_extractors[k].loadParams(f"params/params-{k}.yaml")

    def __getitem__(self, key: str) -> RadiomicsFeatureExtractor:
        return self.dataset[key]

    def __iter__(self):
        for k in self.dataset:
            yield k

    def resample_to_target(
        self, image: sitk.Image, reference_image: sitk.Image
    ) -> sitk.Image:
        return sitk.Resample(
            image,
            reference_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
        )

    def process_masks(
        self, masks: list[str], reference_image: sitk.Image
    ) -> sitk.Image:
        return [
            sitk.Cast(
                self.resample_to_target(mask, reference_image) > 0.5,
                sitk.sitkUInt32,
            )
            for mask in masks
        ]

    def __call__(self, k: str):
        out_path = f"{FEATURE_DIR}/{k}.json"
        if os.path.exists(out_path):
            return k, "already present"
        masks = [sitk.ReadImage(mask) for mask in self[k]["mask"]]
        features_curr = []
        for idx in self.feature_extractors:
            if idx in self[k]:
                image = sitk.ReadImage(self[k][idx]["path"])
                curr_masks = self.process_masks(masks, image)
                mask_idx = 0
                for mask in curr_masks:
                    if sitk.GetArrayFromImage(mask).sum() < 10:
                        break
                    features = self.feature_extractors[idx].execute(image, mask)
                    features = {
                        k: (
                            features[k].tolist()
                            if isinstance(features[k], np.ndarray)
                            else features[k]
                        )
                        for k in features
                    }
                    features["identifier"] = k
                    features["phase"] = idx
                    features["mask_idx"] = mask_idx
                    features_curr.append(features)
                    mask_idx += 1
        with open(out_path, "w") as o:
            json.dump(features_curr, o)
        return k, "success"

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from multiprocessing import Pool

    os.makedirs(FEATURE_DIR, exist_ok=True)

    n_workers = 8

    with open("dataset.json") as o:
        dataset = json.load(o)

    feature_extractors = FeatureExtractors(dataset)

    with Pool(n_workers) as p:
        all_keys = list(feature_extractors.dataset.keys())
        iterator = p.imap_unordered(feature_extractors, all_keys, chunksize=1)
        with tqdm(iterator, total=len(feature_extractors)) as pbar:
            for key, message in pbar:
                pbar.set_description(f"Current patient: {key}")
