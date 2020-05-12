# Code files:
run_ssp_ut.py: Point of entry to train and test Universal Transformer. Download full datasets from the Google Drive link below or run on sample versions of the Redundancy-weighted Q8 data included with the repository.

Example usage:


conv_universal_transformer.py: Encoder block and transition function for convolutional augmentations to UT

model_config.yaml: Model parameters

# Models
cb513_weights.h5: Trained on CB5928 dataset and tested on CB513 dataset

weightq13_weights.h5: Trained and tested on Redundancy-weighted Q13 dataset.

weightq8_weights.h5: Trained and tested on Redundancy-weighted Q13 dataset.

# Keras transformer
All code in this directory is from the [Keras-Transformer](https://github.com/kpot/keras-transformer) library, which we used to build our Universal Transformer model. For easier installation/setup we have included the files required for our project in this directory as well as the code license.

# Data
Contains sample data to train model for Q8 prediction.

Full, processed data for this model can be found at: https://drive.google.com/file/d/1QMBpb0SmDoMTK0dhcfcx8J1dxpT7V4iO/view?usp=sharing

Data files should be copied to Data directory for use.

Raw data files:
CB513/5926 datasets found at: http://www.princeton.edu/~jzthree/datasets/ICML2014/
Redundancy-Weighted dataset found at :http://meshi1.cs.bgu.ac.il/rw/
Use command 'wget --continue --tries=0 -O <dir_name>.tar.gz http://meshi1.cs.bgu.ac.il/rw_datasets_download'
Unzip files and run script 'utils/redundancy_weighted_preprocess.py' to obtain train, validate, test files for DSSP and STR2 standards

# Utils
Data processing scripts

# Results
Predicted sequences per dataset organized into CSV files.

Format is (index, AA sequence, ground truth structure, predicted structure)
