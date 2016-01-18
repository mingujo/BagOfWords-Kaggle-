"""
Purpose:
-----------------------------------------------------------------------------------
- Convolutional Neural Network
- For the model, we actually use published pre-trained vectors trained on part of 
Google News dataset. The model contains 300-dimensional vectors for 3 million words
 and phrases. (zip file is 1.5GB)
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import numpy as np
import pandas as pd
import cPickle
from preprocess_conv_neural import *
from collections import defaultdict
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta
from keras.constraints import unitnorm, maxnorm
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU


train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)
# Downloaded w2v trained model 
w2v_file = '../../model/GoogleNews-vectors-negative300.bin'

### Load preprocessed train and test data
reviews, vocab = preprocess_train_test_review(train, test)
max_l = np.max(pd.DataFrame(reviews)['num_words'])
print "data loaded"

### Load Google's word2vec model from google news data
w2v = load_bin_vec(w2v_file, vocab)
print "w2v loaded"

### Create the mapping for each word's 300 feature vectors
W, word_idx_map = get_W(w2v)

### Store in Pickle format
cPickle.dump([revs, W, word_idx_map, vocab], open('imdb-train-val-test.pickle', 'wb'))
