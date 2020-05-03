DSSP and STR2 protein secondary structure prediction using Universal Transformer neural network.

# Info
Model takes the following features as input:
1) Amino acid sequence (max_len=700)
2) Secondary structure sequence
3) Position Specific Scoring Matrix (PSSM)
4) UniRep 256 unit protein embedding (https://github.com/churchlab/UniRep)

Features 1-4 are used to train the model, features 2-4 are omitted for predictions
Output:
1) Secondary structure sequence

# Results
TODO

# Src
Contains Python code to train Universal Transformer as well as several versions of the trained model for DSSP and STR2 prediction

# Data
Contains sample data to train model.
Redundancy-Weighted dataset found at :http://meshi1.cs.bgu.ac.il/rw/
Use command 'wget --continue --tries=0 -O <dir_name>.tar.gz http://meshi1.cs.bgu.ac.il/rw_datasets_download'
Unzip files and run script 'unpack_data.py' to obtain train, validate, test files for DSSP and STR2 standards
