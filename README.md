# Unet_Application_to_Saturn_Kilometric_Radiation

Python scripts for the study titled 'Image-based classification of intense radio bursts from1 spectrograms: An application to Saturn Kilometric Radiation' published to the [Journal of Geophysical Research](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023JA031926). In this paper we describe how we train a modified U-net architecture for semantic segmentation of Low Frequency Extensions (LFEs) of Saturn Kilometric Radiation (SKR) detected by Cassini/RPWS. We train the model using a catalogue found at (https://doi.org/10.5281/zenodo.7895766).

The batch file (start.cmd) found in the project root can be run to download the spacecraft trajectory data and the training set listed above. The Cassini radio data can be found [here](https://doi.org/10.25935/zkxb-6c84). It needs to be concatenated from daily to yearly files and normalised to 1 astronomical unit (AU) before it can be used in the model. I'm working on incorporating this into the current scripts.

## 'data/input_data' should have the following files:
### LFE files for training:
  'SKR_LFEs.json'\
  
### radio data:
  'SKR_2004_CJ.sav'\
  ...\
  'SKR_2017_001-258_CJ.sav'
  
### Trajectory data: 
  '2004_FGM_KRTP_1M.csv'\
  ...\
  '2017_FGM_KRTP_1M.csv'\
  '2004_FGM_KSM_1M.csv'\
  ...\
  '2017_FGM_KSM_1M.csv'

## Data_prep.py
Run this script to generate the training data which is saved to 'output_data/train' and 'output_data/test' respectively.

## Train_model.py
Run this script for model training. We implement MLOps with MlFlow, storing the runs in the 'mlruns' folder. 

Within the 'Scripts' folder, there are more folders containing the scripts for generating predictions for the rest of the mission and making figures used in the paper. I'll be updating these to make it possible to run from the root folder.
