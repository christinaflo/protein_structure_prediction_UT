"""
Script to train, evaluate, and perform predictions with a convolution-augmented Universal Transformer model.
Contains CLI that allows user to specify which action to take (train, predict, metrics) for one of three available datasets:
    1. CullPDB 5926 (cb513)
    2. Redundancy-weighted PDB Q8 (weightq8)
    3. Redundancy-weighted PDB Q13 (weightq13)
"""

import os
import click
import yaml
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime

import tensorflow as tf
from keras import backend as K
from keras.layers import Embedding, Dense, TimeDistributed, concatenate, SeparableConv1D, Conv1D, Dropout
from keras.models import Model, Input
from sklearn.metrics import classification_report, confusion_matrix

from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, LayerNormalization
from conv_universal_transformer import ConvUniversalTransformerBlock

from utils.preprocess import DataPreprocessor

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RESULTS_PATH = os.path.join(FILE_PATH, 'results')
WEIGHTS_PATH = os.path.join(FILE_PATH, 'models')

CONFIG = yaml.safe_load(open('model_config.yaml', "r"))

MAXLEN_SEQ = 700
PSSM_SIZE = 22

METRICS_TEMPLATE = '''
# ------------------------------------------------
# Confusion Matrix & Metrics                 
# ------------------------------------------------
{}

{}
'''


#--------------------------------------------------------------------------------------------
# NOTE: The following two functions were taken from cu-ssp. Accuracy function was used in
# order to compare results using the same metric, the other is a simple utility function.
# https://github.com/idrori/cu-ssp
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s
#--------------------------------------------------------------------------------------------


def prot_universal_transformer_model(input_size, input_dim, num_labels):
    """Builds the model."""

    model_params = CONFIG['model_params']
    embed_size = model_params['embed_size']

    seq_in = Input(shape=(input_size,))
    pssm_in = Input(shape=(input_size, PSSM_SIZE))
    embedded_seq = Embedding(input_dim=input_dim,
                             output_dim=embed_size,
                             input_length=input_size)(seq_in)

    features = concatenate([embedded_seq, pssm_in], axis=-1)
    features = LayerNormalization()(features)

    # Lower dimensionality, must be divisible by number of attention heads
    conv1 = Conv1D(filters=embed_size,
                   strides=1,
                   padding='same',
                   kernel_size=1)(features)
    ut_x = LayerNormalization()(conv1)

    # Define positional/timestep embedding layer, UT block, and ACT
    transformer_layers = model_params['transformer_layers']
    coordinate_embedding_layer = TransformerCoordinateEmbedding(transformer_layers,
                                                                name='coordinate_embedding')
    act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = ConvUniversalTransformerBlock(
                name='universal_transformer',
                num_heads=model_params['attention_heads'],
                filter_size=embed_size,
                residual_dropout=model_params['residual_dropout'],
                attention_dropout=model_params['attention_dropout'],
                conv_dropout=model_params['conv_dropout'],
                # For bi-directional attention, disable masking
                use_masking=model_params['mask_attention'])

    # Run UT blocks
    act_output = ut_x
    for i in range(transformer_layers):
         ut_x = coordinate_embedding_layer(ut_x, step=i)
         ut_x = transformer_block(ut_x)
         ut_x, act_output = act_layer(ut_x)

    act_layer.finalize()
    ut_x = act_output

    # Predict labels
    y = TimeDistributed(Dense(num_labels, activation="softmax"))(ut_x)

    model = Model([seq_in, pssm_in], y)
    model.summary()
    model.compile(optimizer=model_params['optimizer'],
                  loss="categorical_crossentropy",
                  metrics=["accuracy", accuracy])
    return model


def display_metrics(results):
    """
    Given a CSV file of the actual and expected results, display the
    confusion matrix along with precision, recall, f1-score, and support.
    """

    mismatched_len_df = results[results['actual'].apply(len) != results['expected'].apply(len)]

    # If the lengths of the two sequences do not match, exclude from the report
    # since it is unclear how they align
    if not mismatched_len_df.empty:
        click.echo(mismatched_len_df)
        results.drop(index=mismatched_len_df.index.tolist(), inplace=True)

    test_actual_seqs = results['actual'].tolist()
    test_expected_seqs = results['expected'].tolist()

    exp = np.array([c for i, s in enumerate(test_expected_seqs) for c in s]).flatten()
    act = np.array([c for i, s in enumerate(test_actual_seqs) for c in s]).flatten()

    # Get confusion matrix
    cm = confusion_matrix(exp, act)
    cm_df = pd.DataFrame(cm, index= sorted(list(set(exp))), columns=sorted(list(set(exp))))
    cm_tbl = tabulate(cm_df, headers='keys', tablefmt='psql')
    # Get metrics
    cr = classification_report(exp, act, zero_division=0)

    click.echo(METRICS_TEMPLATE.format(cm_tbl, cr))

@click.group()
def cli():
    pass

@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
def metrics(results_file):
    """
    Display metrics for train/predict CSV results file.
    """
    results = pd.read_csv(results_file)
    display_metrics(results)

@cli.command()
@click.argument('mode', type=click.Choice(['cb513', 'weightq8', 'weightq13']))
@click.argument('weights_file', type=click.Path(exists=True))
def predict(mode, weights_file):
    """
    Predict with model given a specified dataset mode and the path to an h5 file with
    the model weights.
    """

    # Get test data and predict
    preprocessor = DataPreprocessor(mode=mode, maxlen_seq=MAXLEN_SEQ)
    test = preprocessor.get_test_data()
    model = prot_universal_transformer_model(MAXLEN_SEQ, preprocessor.num_words, preprocessor.num_labels)

    model.load_weights(weights_file)

    # Evaluate as well
    acc = model.evaluate([test.sequence, test.profile], test.labels)
    click.echo(f'Test Set Evaluation:\n val_loss:{acc[0]}, val_acc:{acc[1]}, val_accuracy:{acc[2]}')

    predicted = model.predict([test.sequence, test.profile])

    # Verify the number of inputs and outputs match
    assert(len(test.sequence) == len(predicted))

    actual = [str(onehot_to_seq(pred, preprocessor.target_token_map)).upper()
              for pred in predicted]

    # Write results to dir and display metrics
    test.seq_df['actual'] = actual
    daterun = datetime.now().strftime('%Y%m%dT%H%M')
    test.seq_df.to_csv(os.path.join(RESULTS_PATH, f'{mode}_test_{daterun}.csv'))
    display_metrics(test.seq_df)

@cli.command()
@click.argument('mode', type=click.Choice(['cb513', 'weightq8', 'weightq13']))
@click.option('--validate-test', is_flag=True, help='Use test set as validation.')
def train(mode, validate_test):
    """
    Train model given a specified dataset mode. Outputs the model weights,
    prediction results, and displays metrics.
    """

    run_config = CONFIG['run_params']
    preprocessor = DataPreprocessor(mode=mode, maxlen_seq=MAXLEN_SEQ)

    train = preprocessor.get_train_data()
    test = preprocessor.get_test_data()
    validate = test if validate_test else preprocessor.get_validate_data()

    model = prot_universal_transformer_model(MAXLEN_SEQ, preprocessor.num_words,
                                             preprocessor.num_labels)

    # Train model for 15 epochs with batch size 16
    model.fit([train.sequence, train.profile], train.labels,
              batch_size=run_config['batch_size'], epochs=run_config['epochs'],
              validation_data=([validate.sequence, validate.profile], validate.labels),
              verbose=1)

    # Evaluate test set
    acc = model.evaluate([test.sequence, test.profile], test.labels)
    click.echo(f'Test Set Evaluation:\nval_loss:{acc[0]}, val_acc:{acc[1]}, val_accuracy:{acc[2]}')

    predicted = model.predict([test.sequence, test.profile])

    # Verify the number of inputs and outputs match
    assert(len(test.sequence) == len(predicted))

    actual = [str(onehot_to_seq(pred, preprocessor.target_token_map)).upper()
              for pred in predicted]

    # Write results to dir and display metrics
    test.seq_df['actual'] = actual
    daterun = datetime.now().strftime('%Y%m%dT%H%M')
    test.seq_df.to_csv(os.path.join(RESULTS_PATH, f'{mode}_test_{daterun}.csv'))
    model.save_weights(os.path.join(WEIGHTS_PATH, f'{mode}_weights_{daterun}.h5'))
    display_metrics(test.seq_df)

if __name__ == '__main__':
    cli()
