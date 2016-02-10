
# coding: utf-8

# # Custom cifar-100 conv net with Caffe in Python (Pycaffe)
# 
# Here, I train a custom convnet on the cifar-100 dataset. I will try to build a new convolutional neural network architecture. It is a bit based on the NIN (Network In Network) architecture detailed in this paper: http://arxiv.org/pdf/1312.4400v3.pdf. 
# 
# I mainly use some convolution layers, cccp layers, pooling layers, dropout, fully connected layers, relu layers, as well ass sigmoid layers and softmax with loss on top of the neural network. 
# 
# My code, other than the neural network architecture, is inspired from the official caffe python ".ipynb" examples available at: https://github.com/BVLC/caffe/tree/master/examples.
# 
# Please refer to https://www.cs.toronto.edu/~kriz/cifar.html for more information on the nature of the task and of the dataset on which the convolutional neural network is trained on.

# ## Dynamically download and convert the cifar-100 dataset to Caffe's HDF5 format using code of another git repo of mine.
# More info on the dataset can be found at http://www.cs.toronto.edu/~kriz/cifar.html.

# In[2]:

get_ipython().run_cell_magic(u'time', u'', u'\n!rm download-and-convert-cifar-100.py\nprint("Getting the download script...")\n!wget https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-100.py\nprint("Downloaded script. Will execute to download and convert the cifar-100 dataset:")\n!python download-and-convert-cifar-100.py')


# ## Build the model with Caffe. 

# In[3]:

import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P


# In[4]:

def cnn(hdf5, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label_fine)
    return n.to_proto()

def cnn_huge(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3)
    
    n.conv1 = L.Convolution(n.data, kernel_size=4, num_output=64, weight_filler=dict(type='xavier'))
    n.cccp1a = L.Convolution(n.conv1, kernel_size=1, num_output=42, weight_filler=dict(type='xavier'))
    n.relu1a = L.ReLU(n.cccp1a, in_place=True)
    n.cccp1b = L.Convolution(n.relu1a, kernel_size=1, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.cccp1b, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, in_place=True)
    n.relu1b = L.ReLU(n.drop1, in_place=True)
    
    n.conv2 = L.Convolution(n.relu1b, kernel_size=4, num_output=42, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, in_place=True)
    n.relu2 = L.ReLU(n.drop2, in_place=True)
    
    n.conv3 = L.Convolution(n.relu2, kernel_size=2, num_output=64, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    n.relu3 = L.ReLU(n.pool3, in_place=True)
    
    n.ip1 = L.InnerProduct(n.relu3, num_output=768, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.ip1, in_place=True)
    
    n.ip_c = L.InnerProduct(n.sig1, num_output=20, weight_filler=dict(type='xavier'))
    n.accuracy_c = L.Accuracy(n.ip_c, n.label_coarse)
    n.loss_c = L.SoftmaxWithLoss(n.ip_c, n.label_coarse)
    
    n.ip_f = L.InnerProduct(n.sig1, num_output=100, weight_filler=dict(type='xavier'))
    n.accuracy_f = L.Accuracy(n.ip_f, n.label_fine)
    n.loss_f = L.SoftmaxWithLoss(n.ip_f, n.label_fine)
    
    return n.to_proto()
    
with open('cnn_train.prototxt', 'w') as f:
    f.write(str(cnn('cifar_100_caffe_hdf5/train.txt', 100)))
    
with open('cnn_test.prototxt', 'w') as f:
    f.write(str(cnn('cifar_100_caffe_hdf5/test.txt', 120)))


# ## Load and visualise the untrained network's internal structure and shape
# The network's structure (graph) visualisation tool of caffe is broken in the current release. We will simply print here the data shapes. 

# In[3]:

caffe.set_mode_gpu()
solver = caffe.get_solver('cnn_solver_rms.prototxt')


# In[2]:

solver


# In[1]:

print("Layers' features:")
[(k, v.data.shape) for k, v in solver.net.blobs.items()]


# In[ ]:

print("Parameters and shape:")
[(k, v[0].data.shape) for k, v in solver.net.params.items()]


# ## Solver's params
# 
# The solver's params for the created net are defined in a `.prototxt` file. 
# 
# Notice that because `max_iter: 100000`, the training will loop 2 times on the 50000 training data. Because we train data by minibatches of 100 as defined above when creating the net, there will be a total of `100000*100/50000 = 200` epochs on some of those pre-shuffled 100 images minibatches.
# 
# We will test the net on `test_iter: 100` different test images at each `test_interval: 1000` images trained. 
# ____
# 
# Here, **RMSProp** is used, it is SDG-based, it converges faster than a pure SGD and it is robust.
# ____

# In[ ]:

get_ipython().system(u'cat cnn_solver_rms.prototxt')


# ## Alternative way to train directly in Python
# Since a recent update, there is no output in python by default, which is bad for debugging. 
# Skip this cell and train with the second method shown below if needed. It is commented out in case you just chain some `shift+enter` ipython shortcuts. 

# In[ ]:

# %%time
# solver.solve()


# ## Train by calling caffe in command line
# Just set the parameters correctly. Be sure that the notebook is at the root of the ipython notebook server. 
# You can run this in an external terminal if you open it in the notebook's directory. 
# 
# It is also possible to finetune an existing net with a different solver or different data. Here I do it, because I feel the net could better fit the data. 

# In[9]:

get_ipython().run_cell_magic(u'time', u'', u'!$CAFFE_ROOT/build/tools/caffe train -solver cnn_solver_rms.prototxt')


# Caffe brewed. 
# ## Test the model completely on test data
# Let's test directly in command-line:

# In[10]:

get_ipython().run_cell_magic(u'time', u'', u'!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_150000.caffemodel -iterations 83')


# ## The model achieved near 58% accuracy on the 20 coarse labels and 47% accuracy on fine labels.
# This means that upon showing the neural network a picture it had never seen, it will correctly classify it in one of the 20 coarse categories 58% of the time or it will classify it correctly in the fine categories 47% of the time right, and ignoring the coarse label. This is amazing, but the neural network for sure could be fine tuned with better solver parameters. 
# 
# It would  be also possible to have two more loss layers on top of the existing loss, to recombine the predictions made and synchronize with the fact that coarse and fine labels influence on each other and are related.
# 
# This neural network training could be compared to the results listed here: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#494c5356524332303132207461736b2031
# 
# Let's convert the notebook to github markdown:

# In[11]:

get_ipython().system(u'jupyter nbconvert --to markdown custom-cifar-100.ipynb ')
get_ipython().system(u'mv custom-cifar-100.md README.md')

