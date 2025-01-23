# How to Develop Your First Radiomics Signature on CT Images

This script has been forked from (https://github.com/CCIG-Champalimaud/waw-tace-radiomics/tree/master?tab=readme-ov-file) and has been modified to fit radiologist residents for the purpose of learning and training.

## Citation

Code: If you use this code in your work, please cite the original repository by CCIG-Champalimaud: https://github.com/CCIG-Champalimaud/waw-tace-radiomics/tree/master?tab=readme-ov-file
Data set:  [WAW-TACE dataset](https://www.google.com/search?q=waw-tace&oq=WAW-tace&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARBFGD0yBggCEEUYOzIGCAMQRRg8MgYIBBBFGD3SAQc4OTNqMGoxqAIAsAIA&sourceid=chrome&ie=UTF-8). 
This dataset contains hepatocellular carcinoma patients eligible for transarterial chemoembolization therapy, and the task is predicting the progression of each patient.

#Set Colab:
1.new notebook> rename it>save it : it will create a "Colab Notebooks" folder in your google drive 
2. Upload the Dataset to a dedicated folder in Google Drive (e.g., Colab Notebooks/WAW-TACE-radiomics).

# Clone the Repository by copying and pasting this line code into the code line in your notebook:
!git clone https://github.com/MaryamHashim/waw-tace-radiomics.git
#Change directory to the repository:
%cd /content/waw-tace-radiomics
# Install uv and Required Libraries
!pip install uvicorn uv SimpleITK tqdm
# Mount Google Drive:
from google.colab import drive 
drive.mount('/content/drive')
# Give all permission
# Set the Dataset Path: 
DATASET_PATH = "/content/drive/My Drive/Colab Notebooks/WAW-TACE-radiomics"
#Change to the dataset directory:
%cd /content/drive/My\ Drive/Colab\ Notebooks/WAW-TACE-radiomics
#Check that the patients data are there
!ls /content/drive/My\ Drive/Colab\ Notebooks/WAW-TACE-radiomics
#Should find  CT patients data and masks..etc
# Get Dataset 
!uv run scripts/get-dataset.py
# Set Parameters:
!uv run scripts/set-params.py
# Extract Features:
!uv run scripts/extract-features.py
# Compile Dataset:
!uv run scripts/compile-dataset.py

## Notebooks
The notebooks in `notebooks` are illustrative and structured as follows:
1. `1_image_preprocessing_and_radiomic_feature_extraction.ipynb` does all tasks related to dataset processing for the entire dataset and radiomic feature extraction for a single case
2. `2_model_training.ipynb` does all the model training and tuning
#
