DSSP and STR2 protein secondary structure prediction using Universal Transformer neural network.

# INFO

The model was tested using the following software configuration:

Ubuntu 16.04.6

CUDA 9.0

Python 3.6

GPU: Nvidia K80

Nvidia Driver: 440.64.00

Python packages used can be found in requirements.txt

Input:
1) Amino acid sequence (max_len=700)
2) Secondary structure sequence
3) Position Specific Scoring Matrix (PSSM)

Output:
1) Secondary structure sequence

# ssp_universal_transformer
Model code, trained model, data
# Example runs
Tensorflow output of trained model on test datasets. Includes raw counts for confusion matrix as well as accuracy scores.

