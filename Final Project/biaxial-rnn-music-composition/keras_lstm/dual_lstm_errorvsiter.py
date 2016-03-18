from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
import numpy as np
import random
import sys
from midi.utils import midiread, midiwrite
import os

import csv


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


MIDI_RANGE = (21, 109)
PIANO_RANGE = MIDI_RANGE[1] - MIDI_RANGE[0]
MEASURES = 4
TICKS_PER_MEASURE = 16
STEP = 9
EPOCHS = 1000
BATCH_SIZE = 128
TICKS_PER_INPUT = MEASURES*TICKS_PER_MEASURE
GEN_LENGTH = TICKS_PER_INPUT*8
DT = 0.3
    
if __name__ == '__main__':
  #names = ["very_easy", "easy", "medium", "hard", "harder", "fulldata"]
  names = ["fulldata"]

  # Number of iterations to do for each step of the curriculum training, harder has many more
  # it has >150 files
  iters_by_difficulty = {"very_easy":300, "easy":500, "medium":500, "hard": 500, "harder":2000, "fulldata":10000}

  history = LossHistory()

  save_dir = './'

  save_file = 'errors_iter_no_curric.csv'

  # with open(save_file, 'w') as csvfile:
  #     fieldnames = ['Epoch', 'Loss']
  #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  #     writer.writeheader()

  print 'building model...'
  model = Sequential()
  model.add(LSTM(512,  return_sequences=True, stateful=True,
                       batch_input_shape=(BATCH_SIZE, TICKS_PER_INPUT, PIANO_RANGE)))
  model.add(Dropout(0.2))
  model.add(LSTM(512,  return_sequences=False))
  model.add(Dense(PIANO_RANGE))
  model.add(Activation('sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  

  params = [(x, int(x.replace('lstm_weights_','').replace('.h5', ''))) for x in os.listdir('./') if 'lstm_weights_' in x]

  # Do curriculum training in order of difficulty
  for difficulty in names:
      #  Using C-normalized data
      if difficulty != 'fulldata':
          data_dir = "../biaxial-rnn-music-composition/C_Curriculum/" + difficulty + "/"
      else:
          data_dir = "../biaxial-rnn-music-composition/music/"

      ### read MIDI
      #data_dir = '../Data/Cirriculum/easy/'
      #data_dir = '../biaxial-rnn-music-composition/music/'

      files = os.listdir(data_dir)
      files = [data_dir + f for f in files if '.mid' in f or '.MID' in f]

      print files 

      dataset = []

      for f in files:
          try:
              dataset.append(midiread(f, MIDI_RANGE, DT).piano_roll)
              print "{} loaded".format(f)
          except IndexError:
              print "Skipping {}".format(f)
              pass

      print np.shape(dataset)

      X = []
      y = []

      for song in dataset:
          for i in range(0, len(song) - TICKS_PER_INPUT, STEP):
              X.append(song[i: i + TICKS_PER_INPUT])
              y.append(song[i + TICKS_PER_INPUT])

      max_samples = (len(X) // BATCH_SIZE) * BATCH_SIZE
      X = X[:max_samples]
      y = y[:max_samples]

      X = np.array(X)
      y = np.array(y)

      print np.shape(X)
      print np.shape(y)


  for param_file, num_iters in params:
      if num_iters in [7, 18, 23, 31, 53, 55, 63, 68]:
          continue

      # Use the following line to load saved weights
      model.load_weights(param_file)
      
      print 'Epoch {}, training...'.format(num_iters)
      model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1, callbacks=[history])

      print "Loss: {}".format(history.losses)

      with open(save_file, 'a') as csvfile:
          fieldnames = ['Epoch', 'Loss']
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writerow({'Epoch': num_iters, 'Loss': history.losses})

          # Clean states again for training
          model.reset_states()
