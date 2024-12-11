# Tutorial on radiomic feature extraction using WAW-TACE data

# Command-line execution

This is a short tutorial on radiomic feature extraction and machine-learning model training using the [WAW-TACE dataset](https://www.google.com/search?q=waw-tace&oq=WAW-tace&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARBFGD0yBggCEEUYOzIGCAMQRRg8MgYIBBBFGD3SAQc4OTNqMGoxqAIAsAIA&sourceid=chrome&ie=UTF-8). This dataset contains hepatocellular carcinoma patients eligible for transarterial chemoembolization therapy, and the task is predicting the progression of each patient.

Shortly, to run the feature extraction pipeline, start by downloading and uncompressing the WAW-TACE dataset from [Zenodo](https://zenodo.org/records/12741586). Then, install `uv` if you haven't done so - this will make running everything that much easier!

Then, run scripts as follows:
1. `uv run scripts/get-dataset.py`
2. `uv run scripts/set-params.py`
3. `uv run scripts/extract-features.py`
4. `uv run scripts/compile-dataset.py`

And voilá! This should yield a set of radiomic features extracted on a per-lesion level in the `data` folder.

## Notebooks

The notebooks in `notebooks` are illustrative and structured as follows:

1. `1_image_preprocessing_and_radiomic_feature_extraction.ipynb` does all tasks related to dataset processing for the entire dataset and radiomic feature extraction for a single case
2. `2_model_training.ipynb` does all the model training and tuning