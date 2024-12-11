import os
import json
import pandas as pd
from glob import glob
from lib import DATA_DIR


def read_json(json_path: str) -> list | dict:
    with open(json_path) as o:
        return json.load(o)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    feature_files = glob("features/*json")

    all_features = {"1": [], "2": []}
    for feature_file in feature_files:
        patient_identifier = feature_file.split(os.sep)[-1].replace(".json", "")
        features = read_json(feature_file)
        if len(features) == 0:
            continue
        for feature in features:
            all_features[feature["phase"]].append(feature)

    for phase in all_features:
        pd.DataFrame(all_features[phase]).to_csv(
            f"{DATA_DIR}/all_data_{phase}.csv", index=False
        )
