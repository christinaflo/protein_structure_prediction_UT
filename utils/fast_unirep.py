import tensorflow as tf
import numpy as np
import pandas as pd
import unirep
import data_utils
from math import ceil
import time

'''
Full credit for run_inference() and DataFrame code goes to GitHub user 'smsaladi' who opened an
issue on the churchlab/UniRep reposity proposing a more efficient implementation of UniRep's
get_rep() function. His modification passes an existing Tensorflow session
to get_rep() and batches the protein sequence inputs.
'''

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


residue_list = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
q8_list = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

def load_augmented_data(npy_path, max_len):
    ''' For Princeton .npy data '''

    data = np.load(npy_path)
    data_reshape = data.reshape(data.shape[0], 700, -1)
    residue_onehot = data_reshape[:,:,0:22]
    residue_q8_onehot = data_reshape[:,:,22:31]
    profile = data_reshape[:,:,35:57]
    residue_array = np.array(residue_list)[residue_onehot.argmax(2)]
    q8_array = np.array(q8_list)[residue_q8_onehot.argmax(2)]

    sequences = list()
    for idx, vec in enumerate(residue_array):
        seq = ''.join(vec[vec != 'NoSeq'])
        pad_length = ceil(len(seq)/100)*100 - len(seq)
        seq = seq + '-'*pad_length # sequences are padded to nearest hundred
        sequences.append((idx, seq))

    sequences = list(zip(*sequences))
    sequences = pd.Series(sequences[1], index=sequences[0], name='seq')
    return sequences

#get_ipython().system('aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/ 64_weights/')
def series_from_file(filename):
    ''' For PDB data '''

    sequences = list()
    with open(filename, 'r') as file:
        for idx, line in enumerate(file):
            seq, ss_seq, pssm = line.strip().split('|')
            seq = seq.replace('-','')
            pad_length = ceil(len(seq)/100)*100 - len(seq)
            seq = seq + '-'*pad_length # sequences are padded to nearest hundred
            sequences.append((idx, seq))
    sequences = list(zip(*sequences))
    sequences = pd.Series(sequences[1], index=sequences[0], name='seq')

    return sequences

def run_inference(filename, dataset):
    if dataset == 'PDB':
        infile='../data/{a}'.format(a=filename)
        f = filename.split('_')
        outfile='../data/{a}_vectors.pkl'.format(a='_'.join([f[0],f[1]]))
        seqs = series_from_file(infile)
    else:
        infile='../data/{a}'.format(a=filename)
        outfile='../data/{a}_vectors.pkl'.format(a=filename[:-4])
        seqs = load_augmented_data(infile, 700)
    batch_size = 1
    # set up babbler object
    b = unirep.babbler64(batch_size=batch_size, model_path="./64_weights")

    # read sequences
    df_seqs = seqs.to_frame()

    # sort by length
    df_seqs['len'] = df_seqs['seq'].str.len()
    df_seqs.sort_values('len', inplace=True)
    df_seqs.reset_index(drop=True, inplace=True)

    df_seqs['grp'] = df_seqs.groupby('len')['len'].transform(lambda x: np.arange(np.size(x))) // batch_size
    print(df_seqs)
    # set up tf session, then run inference
    with tf.Session() as sess:
        unirep.initialize_uninitialized(sess)
        df_calc = df_seqs.groupby(['grp', 'len'], as_index=False, sort=False).apply(lambda d: b.get_rep(d['seq'], sess=sess))

    df_calc.to_pickle(outfile)
    return

def np_to_list(arr):
    arr = arr[0]
    return [arr[i] for i in np.ndindex(arr.shape[:-1])]

aa_to_int = {
    '-':0,
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
}

def aa_seq_to_int(s):
    """Monkey patch to return unknown if not in alphabet
    """
    s = s.strip()
    s_int = [24] + [aa_to_int.get(a, aa_to_int['X']) for a in s] + [25]
    return s_int[:-1]

def get_rep(self, seqs, sess):
    """
    Monkey-patch get_rep to accept a tensorflow session (instead of initializing one each time)
    """
    if isinstance(seqs, str):
        seqs = pd.Series([seqs])

    coded_seqs = [aa_seq_to_int(s) for s in seqs]
    n_seqs = len(coded_seqs)

    if n_seqs == self._batch_size:
        zero_batch = self._zero_state
    else:
        zero = self._zero_state[0][0]
        zero_batch = [zero[:n_seqs,:], zero[:n_seqs, :]]

    final_state_, hs = sess.run(
            [self._final_state, self._output], feed_dict={
                self._batch_size_placeholder: n_seqs,
                self._minibatch_x_placeholder: coded_seqs,
                self._initial_state_placeholder: zero_batch
            })

    final_cell_all, final_hidden_all = final_state_
    avg_hidden = np.mean(hs, axis=1)

    df = seqs.to_frame()
    df['seq'] = seqs
    df['final_hs'] = np_to_list(final_hidden_all)[:n_seqs]
    df['final_cell'] = np_to_list(final_cell_all)[:n_seqs]
    df['avg_hs'] = np_to_list(avg_hidden)[:n_seqs]

    return df
unirep.babbler64.get_rep = get_rep


if __name__ == '__main__':
    train_fname = 'cb5926filtered.npy'
    test_fname = 'cb513.npy'

    PDB_files = ['train_Q13_data.txt', 'test_Q13_data.txt', 'validate_Q13_data.txt', 'train_Q8_data.txt', 'test_Q8_data.txt', 'validate_Q8_data.txt']

    start = time.time()
    run_inference(test_fname, 'CB')
    print(time.time()-start)
