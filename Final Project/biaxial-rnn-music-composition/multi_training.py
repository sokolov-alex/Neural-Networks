import os, random
from datetime import datetime

from midi_to_statematrix import *
from data import *
import cPickle as pickle

import signal

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations

def loadPieces(dirpath, music_path = 'music/pieces.pkl'):

    pieces = {}

    if os.path.exists(music_path):
        print 'Loaded preprocessed notes...'
        return pickle.load(open("music/pieces.pkl", "rb"))
    else:
        for fname in os.listdir(dirpath):
            if fname[-4:] not in ('.mid','.MID'):
                continue

            name = fname[:-4]

            outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
            if len(outMatrix) < batch_len:
                continue

            pieces[name] = outMatrix
            print "Loaded {}".format(name)

        pickle.dump(pieces, open("music/pieces.pkl", "wb"))

    return numpy.array(pieces)

def getPieceSegment(pieces):
    idx = random.choice(range(len(pieces)))
    piece_output = pieces.values()[idx]
    start = random.randrange(0,len(piece_output)-batch_len,division_len)
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start)

    seg_out = piece_output[start:start+batch_len]
    if notes_to_input:
        seg_in = notes_to_input[idx][start:start+batch_len]
    else:
        seg_in = noteStateMatrixToInputForm(seg_out)
    # 128 1980 = 253440   vs   128 78 2 = 19968     vs 128 78 80 = 798720
    return seg_in, seg_out

def getPieceBatch(pieces):
    i,o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return numpy.array(i), numpy.array(o)

def trainPiece(model,pieces,epochs, notesToInput = None, start=0, input_path = 'music/notes_to_input.pkl'):
    '''global notes_to_input
    if not os.path.isfile(input_path):
        notes_to_input = noteStateMatrixToInputForm(pieces.values())
        pickle.dump(notes_to_input,open(input_path, 'wb'))
    else:
        notes_to_input = pickle.load(open(input_path, "rb"))'''
    global notes_to_input
    if notesToInput:
        notes_to_input = notesToInput

    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        if i % 100 == 0:
            print "epoch {}, error={}".format(i,error)
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            t = datetime.now()
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),'output/sample{0}_{1}:{2}'.format(i, t.hour, t.minute))
            pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)
