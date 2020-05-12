# Code files:
run_ssp_ut.py: Point of entry to train and test Universal Transformer. Download full datasets from the Google Drive link below or run on sample versions of the Redundancy-weighted Q8 data included in the data/ directory.

conv_universal_transformer.py: Encoder block and transition function for convolutional augmentations to UT

model_config.yaml: Model parameters

## Usage:
To train the model, run the following command with one of the three listed dataset modes:
```bash
$ python run_ssp_ut.py train [cb513|weightq8|weightq13]
```
This command will also automatically evaluate the test data and output prediction results to the results/ directory in CSV format, as well as the final weights to models/ in h5 file format. Passing the option ```--validate-test``` will allow you to use the test dataset in place of the validation during training if desired.

To predict secondary structure for a specified dataset mode and an existing model, run:
```bash
$ python run_ssp_ut.py predict [cb513|weightq8|weightq13] </path/to/model/weights.h5>
```
This command will also evaluate the results and output predictions to the results/ directory.

To display metrics (confusion matrix, f1-scores, etc.) for an existing results file, run:
```bash
$ python run_ssp_ut.py metrics </path/to/results.csv>
```

### Examples
```bash
$ python run_ssp_ut.py train cb513
```
```bash
$ python run_ssp_ut.py predict weightq13 models/weightq13_weights.h5
```
```bash
$ python run_ssp_ut.py metrics results/cb513_test.csv
```
```bash
$ python run_ssp_ut.py --help
```

# Models
- cb513_weights.h5: Trained on CB5928 dataset and tested on CB513 dataset
- weightq8_weights.h5: Trained and tested on Redundancy-weighted Q8 dataset.
- weightq13_weights.h5: Trained and tested on Redundancy-weighted Q13 dataset.

# Keras transformer
Code from [Keras-Transformer](https://github.com/kpot/keras-transformer) library, which we used to build our Universal Transformer model. For easier installation/setup we have included the files required for our project in this directory as well as the code license.

# Data
Contains sample data to train model for Q8 prediction.

Full, processed data for this model can be found at: https://drive.google.com/file/d/1QMBpb0SmDoMTK0dhcfcx8J1dxpT7V4iO/view?usp=sharing

The directory will contain the following:
- cb5926filtered.npy
- cb5926_validate.npy          
- cb513.npy
- train_Q8_data.txt
- validate_Q8_data.txt
- test_Q8_data.txt
- train_Q13_data.txt
- validate_Q13_data.txt
- test_Q13_data.txt

Data files should be copied to data/ directory for use.

Raw data files:
CB513/5926 datasets found at: http://www.princeton.edu/~jzthree/datasets/ICML2014/

Redundancy-Weighted dataset found at :http://meshi1.cs.bgu.ac.il/rw/

Use command 'wget --continue --tries=0 -O <dir_name>.tar.gz http://meshi1.cs.bgu.ac.il/rw_datasets_download'.
Unzip files and run script 'utils/redundancy_weighted_preprocess.py' to obtain train, validate, test files for DSSP and STR2 standards

Note: if using raw data files, be sure to keep the names consistent with what is used above

# Utils
preprocess.py: Used by run_ssp_ut.py to prepare the specified train/test/validation data to be ingested by the model
redundancy_weighted_preprocess.py: Process Redundancy-Weighted dataset and convert to format consumed by preprocess.py

# Results
Predicted sequences per dataset organized into CSV files.

Format is (index, AA sequence, ground truth structure, predicted structure)
