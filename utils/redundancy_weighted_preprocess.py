from json import load


# File convention:
# X.Y#######.json
# X = 1,2,3 train, validate, test
# Y = 14, 9 Q13, Q8


def unpackData(split, alphabet='Q13'):
    ''' Grabs all unique proteins from across 4 partitions, and writes to file'''
    prefix = {'train': '1', 'validate': '2', 'test': 3}
    vocab = {'Q13': '14', 'Q8': '9'}

    files = [f'./partition{x}/{prefix[split]}.{vocab[alphabet]}.nr100.nr90.uniqueprot_e3_e1_t5.fasta.json' for x in range(1,5)]
    d = list()

    data_file = open(f'{split}_{alphabet}_data.txt', 'w')
    for filename in files:
        with open(filename, 'r') as file:
            data = load(file)

            for pid in data:
                if pid in d:
                    continue

                seq = data[pid]['seq']
                ss_seq = data[pid]['ss_seq']
                pssm = data[pid]['profile_seq']

                data_file.write(f'{seq}|{ss_seq}|{pssm}\n')

    data_file.close()


if __name__ == '__main__':
    splits = ['train', 'validate', 'test']
    vocal = ['Q8', 'Q13']
    for s in splits:
        for v in vocal:
            unpackData(s, v)
