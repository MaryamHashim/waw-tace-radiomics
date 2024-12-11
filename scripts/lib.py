import os

DATA_DIR = "data"
DATASET_PATH = "/big_disk/Datasets/WAW-TACE/"
FEATURE_DIR = "features"


def get_params_path(idx: str | int) -> tuple[str, str]:
    file_name = f"params/params-{idx}.yaml"
    default_file_name = "params/params-template.yaml"
    if os.path.exists(file_name) is False:
        in_file_name = default_file_name
    else:
        in_file_name = file_name
    return in_file_name, file_name
