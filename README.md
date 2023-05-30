# Unet_Application_to_Saturn_Kilometric_Radiation

Python scripts that used in paper that is currently in prep. titled 'Image-based classification of intense radio bursts from1 spectrograms: An application to Saturn Kilometric Radiation'. In this paper we describe how we train a modified U-net architecture for semantic segmentation of Low Frequency Extensions (LFEs) of Saturn Kilometric Radiation (SKR) detected by Cassini/RPWS. We train the model using a catalogue found at (https://doi.org/10.5281/zenodo.7895766).

The configurations file must be updated to users own configuration, it has 3 filepaths. 'input_data' (where the data needed to run the scripts are stored), 'output_data' (where the data generated by the scripts is stored) and then 'model_name' (the name of the folder where the model weights are stored). 
