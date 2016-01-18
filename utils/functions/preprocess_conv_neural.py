"""
Purpose:
-----------------------------------------------------------------------------------
- Build necessary preprocessing functions for original data 
- This is for convolutional neural network
-----------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
import string


def preprocess_train_test_review(labeled_train_review, test_review, train_ratio = 0.8, clean_string=True):
    """
    Loads data and split into train and test sets.

    Parameters
    ----------
    train_review : string
        a movie review (train)

    test_review : string
        a movie review (test)
    
    Returns
    -------
    reviews : a list of dictionaries
        a preprocessed and cleaned review

    vocab : collections.defaultdict

    """
    # Initialize a list 
    reviews = []
    vocab = defaultdict(float)
    # Pre-process train data set (with label)
    for i in xrange(labeled_train_review.shape[0]):
        review = labeled_train_review['review'][i]
        label = labeled_train_review['sentiment'][i]
        # rev = []
        # rev.append(review.strip())
        # If you want to get rid of punctuations
        if clean_string:
            original = tokenize_review(review)
                # ' '.join(rev))
        # If not:
        else:
            original = review.lower()
        # make it faster by making it as set
        words = set(original.split())
        for word in words:
            vocab[word] += 1
        datum  = {'y': label, 
                  'text': original,
                  'num_words': len(original.split()),
                  'split': int(np.random.rand() < train_ratio)}

        reviews.append(datum)
        
    # Pre-process test data set
    for i in xrange(test_review.shape[0]):
        review = test_review['review'][i]
        # rev = []
        # rev.append(review.strip())
        if clean_string:
            original = clean_str(review)
        else:
            original = review.lower()
        words = set(original.split())
        for word in words:
            vocab[word] += 1
        datum  = {'y': -1, 
                  'text': original,
                  'num_words': len(original.split()),
                  'split': -1}
        reviews.append(datum)
        
    return reviews, vocab



def tokenize_review(review):
    """
    Tokenize a review (remove punctuations, put spaces and split)
 
    Parameters
    ----------
    review : string
        a movie review
    
    Returns
    -------
    review : string
        a preprocessed review

    """
    # review = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", review)     
    # review = re.sub(r"\'s", " \'s", review) 
    # review = re.sub(r"\'ve", " \'ve", review) 
    # review = re.sub(r"n\'t", " n\'t", review) 
    # review = re.sub(r"\'re", " \'re", review) 
    # review = re.sub(r"\'d", " \'d", review) 
    # review = re.sub(r"\'ll", " \'ll", review) 
    # review = re.sub(r",", " , ", review) 
    # review = re.sub(r"!", " ! ", review) 
    # review = re.sub(r"\(", " \( ", review) 
    # review = re.sub(r"\)", " \) ", review) 
    # review = re.sub(r"\?", " \? ", review) 
    # review = re.sub(r"\s{2,}", " ", review) 
    review = review.translate(None, string.puncuation)
    review = review.strip().lower()
    return review


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i

    Parameters
    ----------
    review : string
        a movie review
    
    Returns
    -------
    review : string
        a preprocessed review
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
    W[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


