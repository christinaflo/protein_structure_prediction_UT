DSSP and STR2 protein secondary structure prediction using Universal Transformer neural network.

# INFO

The model was tested using the following software configuration:
Ubuntu 16.04
CUDA x.xx
CUDNN x.xx
Python 3.6
Tensorflow x.xx
GPU: Nvidia K80

Python packages used can be found in requirements.txt

Input:
1) Amino acid sequence (max_len=700)
2) Secondary structure sequence
3) Position Specific Scoring Matrix (PSSM)

Output:
1) Secondary structure sequence

# Src
run_ssp_ut.py: Point of entry to train and test Universal Transformer. Download full datasets from the Google Drive link below or run on sample versions of the Redundancy-weighted Q8 data included with the repository.
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
Unzip files and run script 'redundancy_weighted_preprocess.py' to obtain train, validate, test files for DSSP and STR2 standards

# Utils
Data processing scripts

# Example runs
Tensorflow output of trained model on test datasets. Includes raw counts for confusion matrix as well as accuracy scores.

# Results
Predicted sequences per dataset organized into CSV files.
Format is (index, AA sequence, ground truth structure, predicted structure)
