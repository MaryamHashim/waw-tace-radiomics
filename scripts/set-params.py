import os
import json
import numpy as np
import yaml
from math import inf, ceil

from lib import get_params_path

all_spacings = {"0": [], "1": [], "2": [], "3": []}
all_minimums = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
}
with open("dataset.json") as o:
    data = json.load(o)
    for k in data:
        for idx in all_spacings:
            if idx in data[k]:
                all_spacings[idx].append(data[k][idx]["spacing"])
                if all_minimums[idx] > data[k][idx]["minimum"]:
                    all_minimums[idx] = data[k][idx]["minimum"]

for k in all_spacings:
    spacings = np.array(all_spacings[k])
    params_path, out_params_path = get_params_path(k)
    with open(params_path) as o:
        params = yaml.safe_load(o)
    params["setting"]["resampledPixelSpacing"] = np.max(spacings, 0).tolist()
    params["setting"]["resampledPixelSpacing"][-1] = 0
    params["setting"]["voxelArrayShift"] = int(-ceil(all_minimums[idx]))
    with open(out_params_path, "w") as o:
        yaml.dump(params, o)
