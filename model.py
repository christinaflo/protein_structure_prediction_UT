import os
import click
import numpy as np
import pandas as pd
from tabulate import tabulate

import tensorflow as tf
from keras import backend as K
from keras.layers import Embedding, Dense, TimeDistributed, concatenate, SeparableConv1D, Conv1D, Dropout
from keras.models import Model, Input
from sklearn.metrics import classification_report, confusion_matrix

from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, LayerNormalization
from prot_conv_ut import ConvUniversalTransformerBlock

from preprocess import DataPreprocessor

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RESULTS_PATH = os.path.join(FILE_PATH, 'results')
WEIGHTS_PATH = os.path.join(FILE_PATH, 'models')


METRICS_TEMPLATE = '''
# ------------------------------------------------
# Confusion Matrix & Metrics: {}                  
# ------------------------------------------------
{}

{}
'''

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

def prot_universal_transformer_model(input_size, embed_size, input_dim, num_labels):

    seq_in = Input(shape=(input_size,))
    pssm_in = Input(shape=(input_size, 22))
    embedded_seq = Embedding(input_dim=input_dim,
                             output_dim=embed_size,
                             input_length=input_size)(seq_in)

    features = concatenate([embedded_seq, pssm_in], axis=-1)
    features = LayerNormalization()(features)

    conv1 = Conv1D(filters=128,
                   strides=1,
                   padding='same',
                   kernel_size=1)(features)
    ut_x = LayerNormalization()(conv1)

    transformer_layers = 6
    coordinate_embedding_layer = TransformerCoordinateEmbedding(transformer_layers,
                                                                name='coordinate_embedding')
    act_layer = TransformerACT(name='adaptive_computation_time')
    transformer_block = ConvUniversalTransformerBlock(
                name='universal_transformer',
                num_heads=4,
                filter_size=128,
                residual_dropout=0.1,
                attention_dropout=0.1,
                conv_dropout=0.2,
                use_masking=False)

    act_output = ut_x
    for i in range(transformer_layers):
         ut_x = coordinate_embedding_layer(ut_x, step=i)
         ut_x = transformer_block(ut_x)
         ut_x, act_output = act_layer(ut_x)

    act_layer.finalize()
    ut_x = act_output

    y = TimeDistributed(Dense(num_labels, activation="softmax"))(ut_x)

    model = Model([seq_in, pssm_in], y)
    model.summary()
    model.compile(optimizer='Nadam',
                  loss="categorical_crossentropy",
                  metrics=["accuracy", accuracy])
    return model


def display_metrics(mode, results):
    mismatched_len_df = results[results['actual'].apply(len) != results['expected'].apply(len)]

    if not mismatched_len_df.empty:
        click.echo(mismatched_len_df)
        results.drop(index=mismatched_len_df.index.tolist(), inplace=True)

    test_actual_seqs = results['actual'].tolist()
    test_expected_seqs = results['expected'].tolist()

    exp = np.array([c for i, s in enumerate(test_expected_seqs) for c in s]).flatten()
    act = np.array([c for i, s in enumerate(test_actual_seqs) for c in s]).flatten()

    cm = confusion_matrix(exp, act)
    cm_df = pd.DataFrame(cm, index= sorted(list(set(exp))), columns=sorted(list(set(exp))))
    cm_tbl = tabulate(cm_df, headers='keys', tablefmt='psql')
    #cm_tbl = tabulate(cm_df, headers='keys', tablefmt='latex')
    cr = classification_report(exp, act, zero_division=0)
    #cr = classification_report(exp, act, zero_division=0, output_dict=True)
    #cr = tabulate(pd.DataFrame(cr).transpose(), headers='keys', tablefmt='latex')

    click.echo(METRICS_TEMPLATE.format(mode, cm_tbl, cr))

@click.group()
def cli():
    pass

@cli.command()
@click.argument('mode', type=click.Choice(['cb513', 'weightq8', 'weightq13']))
@click.argument('results_file', type=click.Path(exists=True))
def metrics(mode, results_file):
    results = pd.read_csv(results_file)
    display_metrics(mode, results)

@cli.command()
@click.argument('mode', type=click.Choice(['cb513', 'weightq8', 'weightq13']))
@click.argument('weights_file', type=click.Path(exists=True))
def predict(mode, weights_file):
    preprocessor = DataPreprocessor(mode=mode, maxlen_seq=700)
    test = preprocessor.get_test_data()
    model = prot_universal_transformer_model(700, 128, preprocessor.num_words, preprocessor.num_labels)

    model.load_weights(weights_file)
    predicted = model.predict([test.sequence, test.profile])

    assert(len(test.sequence) == len(predicted))

    actual = [str(onehot_to_seq(pred, preprocessor.target_token_map)).upper()
              for pred in predicted]

    test.seq_df['actual'] = actual
    test.seq_df.to_csv(os.path.join(RESULTS_PATH, f'{mode}_test.csv'))
    display_metrics(mode, test.seq_df)

@cli.command()
@click.argument('mode', type=click.Choice(['cb513', 'weightq8', 'weightq13']))
def train(mode):
    preprocessor = DataPreprocessor(mode=mode, maxlen_seq=700)

    train = preprocessor.get_train_data()
    test = preprocessor.get_test_data()

    model = prot_universal_transformer_model(700, 128, preprocessor.num_words,
                                             preprocessor.num_labels)

    model.fit([train.sequence, train.profile], train.labels,
              batch_size=16, epochs=15,
              validation_data=([test.sequence, test.profile], test.labels),
              verbose=1)

    acc = model.evaluate([test.sequence, test.profile], test.labels)
    click.echo(acc)

    predicted = model.predict([test.sequence, test.profile])

    assert(len(test.sequence) == len(predicted))

    actual = [str(onehot_to_seq(pred, preprocessor.target_token_map)).upper()
              for pred in predicted]

    test.seq_df['actual'] = actual
    test.seq_df.to_csv(os.path.join(RESULTS_PATH, f'{mode}_test.csv'))
    model.save_weights(os.path.join(WEIGHTS_PATH, f'{mode}_weights.h5'))
    display_metrics(mode, test.seq_df)

if __name__ == '__main__':
    cli()
