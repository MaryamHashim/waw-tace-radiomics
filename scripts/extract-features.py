import os
import json
import numpy as np
import SimpleITK as sitk
from radiomics.featureextractor import RadiomicsFeatureExtractor
from radiomics import setVerbosity
from tqdm import tqdm

feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)
setVerbosity(60)


class FeatureExtractors:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.feature_extractors = {
            "0": RadiomicsFeatureExtractor(),
            "1": RadiomicsFeatureExtractor(),
            "2": RadiomicsFeatureExtractor(),
            "3": RadiomicsFeatureExtractor(),
        }

        for k in self.feature_extractors:
            self.feature_extractors[k].loadParams(f"params/params-{k}.yaml")

    def __getitem__(self, key: str) -> RadiomicsFeatureExtractor:
        return self.dataset[key]

    def __iter__(self):
        for k in self.dataset:
            yield k

    def resample_to_target(
        self, image: sitk.Image, target_image: sitk.Image
    ) -> sitk.Image:
        return sitk.Resample(
            image, target_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0
        )

    def process_masks(
        self, masks: list[str], reference_image: sitk.Image
    ) -> sitk.Image:
        mask = sum(
            [
                sitk.Cast(
                    self.resample_to_target(mask, reference_image),
                    sitk.sitkUInt32,
                )
                for mask in masks
            ]
        ) > 0.5
        mask = sitk.Cast(mask, sitk.sitkUInt32)
        return mask

    def __call__(self, k: str):
        out_path = f"{feature_dir}/{key}.json"
        if os.path.exists(out_path):
            print(f"{out_path} already present, skipping")
            return
        masks = [sitk.ReadImage(mask) for mask in self[k]["mask"]]
        features_curr = []
        for idx in self.feature_extractors:
            if idx in self[k]:
                image = sitk.ReadImage(self[k][idx]["path"])
                curr_mask = self.process_masks(masks, image)
                features = self.feature_extractors[idx].execute(
                    image, curr_mask
                )
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
        with open(out_path, "w") as o:
            json.dump(features_curr, o)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    with open("dataset.json") as o:
        dataset = json.load(o)

    feature_extractors = FeatureExtractors(dataset)

    for key in tqdm(feature_extractors):
        feature_extractors(key)
