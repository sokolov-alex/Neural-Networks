import cPickle as pickle
import gzip
import os

import numpy as np

from RBM import rbm_reconstruction
from midi_to_statematrix import *

import multi_training
import model
import numpy as np


def gen_adaptive(m, pcs, times, keep_thoughts=False, name="final", rbm = False):
    xIpt, xOpt = map(lambda x: np.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))
    all_outputs = [xOpt[0]]
    if keep_thoughts:
        all_thoughts = []
    m.start_slow_walk(xIpt[0])
    cons = 1
    for time in range(multi_training.batch_len * times):
        resdata = m.slow_walk_fun(cons)
        nnotes = np.sum(resdata[-1][:, 0])
        if nnotes < 2:
            if cons > 1:
                cons = 1
            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3
        all_outputs.append(resdata[-1])
        if keep_thoughts:
            all_thoughts.append(resdata)
    if rbm:
        from RBM import rbm_reconstruction
        rbm_outputs = rbm_reconstruction(np.array(pcs.values()), np.array(all_outputs), window_len = 16,
                                         learning_rate=0.1, training_epochs=15, batch_size=20, n_hidden=1500)
        noteStateMatrixToMidi(np.array(rbm_outputs), 'output/' + name + ' rbm')
    noteStateMatrixToMidi(np.array(all_outputs), 'output/' + name)
    if keep_thoughts:
        pickle.dump(all_thoughts, open('output/' + name + '.p', 'wb'))


def fetch_train_thoughts(m, pcs, batches, name="trainthoughts"):
    all_thoughts = []
    for i in range(batches):
        ipt, opt = multi_training.getPieceBatch(pcs)
        thoughts = m.update_thought_fun(ipt, opt)
        all_thoughts.append((ipt, opt, thoughts))
    pickle.dump(all_thoughts, open('output/' + name + '.p', 'wb'))

if __name__ == '__main__':
    path = 'C_music'
    batches = 7000
    batches_old = 0

    pieces = multi_training.loadPieces(dirpath = path)

    rbm_outputs = rbm_reconstruction(np.array(pieces.values()), np.array(pieces.values()), window_len = 16,
                                         learning_rate=0.1, training_epochs=15, batch_size=20, n_hidden=1500)

    m = model.Model([300, 300], [100, 50], dropout=0.5)

    m.learned_config = pickle.load(open("output/final_learned_config_7000.p", "rb"))

    gen_adaptive(m,pieces,10,name="composition_{0}".format(batches+batches_old), rbm=True)

    print 'Training {0}+{1} batches on {2}'.format(batches,batches_old, path)

    multi_training.trainPiece(m, pieces, [batches, batches_old])#, notes_to_input = None)
    pickle.dump(m.learned_config, open("output/final_learned_config_{0}.p".format(batches+batches_old), "wb"))