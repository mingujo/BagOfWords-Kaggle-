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
            original = tokenize_review(review)
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
    review = review.translate(None, string.punctuation)
    review = review.strip().lower()
    return review

def load_bin_vec(fname, vocab):
    """ 
    Loads 300x1 word vecs from Google word2vec
    (because word vector dimensionality that the tutorial used is 300)
    The downloaded model is in some raw binary data.
    We need to convert into a string so we can use
    (Straightly brought)

    Parameters
    ----------
    fname : directory
        directory where the bin file is

    vocab : collections.defaultdict
        a movie review (test)
    
    Returns
    -------
    word_vecs : dictionary

    
    """
    word_vecs = {}
    # load the google news model
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs


def get_word_matrix(word_vecs, k=300):
    """ 
    Get word matrix. W[i] is the vector for word indexed by i
    Each word in matrix is a number
    word_idx_map is a mapping for a word from an index
    Eventually, you get a word feature vector from a word

    Parameters
    ----------
    word_vecs : string
        a movie review (train)

    k : int
    
    Returns
    -------
    word_matrix : np.array (matrix)
        a preprocessed and cleaned review

    word_idx_map : collections.defaultdict

    
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    word_matrix = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
    word_matrix[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        word_matrix[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return word_matrix, word_idx_map



def get_idx_from_sentences(sentence, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in xrange(pad):
        x.append(0)
    words = sentence.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x



def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l, kernel_size)
        sent.append(rev['y'])
        if rev['split'] == 1:
            train.append(sent)
        elif rev['split'] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=np.int)
    val = np.array(val, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return [train, val, test]


