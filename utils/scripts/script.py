"""
Purpose:
-----------------------------------------------------------------------------------
- Preprocess the review data
- Build a model
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup # to extract text only
from clean_data import *
from avgFeatureVec import *
from sklearn.feature_extraction.text import CountVectorizer
import nltk.data
from gensim.models import word2vec
from gensim.models import Word2Vec #(for loading #Word2Vec object)


############################ Load and Preprocess train data
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)
# we now use this because word2vec doesn't need labels. this has additional 50000 reviews
unlabeled_train = pd.read_csv( "../../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )


# nltk.download() # find and download 'stopwords' (meaningless words such as 'a', 'is', 'each', 'here')

# To train Word2Vec it is better not to remove stop words because the algorithm relies on the 
# broader context of the sentence in order to produce high-quality word vectors

# Get punctuation tokenizer for senetence splitting
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# --> review_to_sentences
# split a pragraph into sentences
sentences = []
print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)


############################ TRAIN 

import logging
# For specifying output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

print "Training model..."
# Use Google's word2vec API to train (hierarchial softmax, k-means clustering)
model = word2vec.Word2Vec(sentences, workers=4, size=300, min_count = 40, \
            window = 10, sample = 1e-3)
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)

## test
print "Hey model, what is the one that does not match in (france, england, berlin, germany)" 
print(model.doesnt_match("france england berlin germany".split()))


############################ Now we have a trained model with some semantic understanding of words
# The model is stored in a numpy array
model = Word2Vec.load("300features_40minwords_10context")
print "The shape of the Word2Vec model:"
print(model.syn0.shape)
print "The number of rows in syn0 is the number of words in the model's vocabulary, \
	and the number of columns corresponds to the size of the feature vector"
