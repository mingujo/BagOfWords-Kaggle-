"""
Purpose:
-----------------------------------------------------------------------------------
- Clean Data
-----------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup # to extract text only
import re
import nltk
from nltk.corpus import stopwords


def review_to_word(review):
    """ Return the string of words from a raw imdb review 
 
    Parameters
    ----------
    review : string
        a movie review
    
    Returns
    -------
    meaningful_words : string
        a preprocessed and cleaned review

    """
    # Get text only
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # searching in a set rather than a list is faster in python
    stops = set(stopwords.words("english"))
    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # Join the words back into one string
    return( " ".join( meaningful_words ))

def review_to_wordlist(review, remove_stopwords=False):
    """ Return the list of words from a raw imdb review 
 
    Parameters
    ----------
    review : string
        a movie review

    remove_stopwords : boolean
        whether to remove stopwords
    
    Returns
    -------
    words : list of strings
        

    """
    # Get text only
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                                   
    # searching in a set rather than a list is faster in python
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """ Return a splited review into parsed sentences
 
    Parameters
    ----------
    review : string
        a movie review

    tokenizer : NLTK tokenizer
        required to split the paragraph into sentences

    remove_stopwords : boolean
        whether to remove stopwords
    
    Returns
    -------
    sentences : list of lists
        each sentence(list) is a list of words


    """
    raw_sentences = tokenizer.tokenize(review.strip())

    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence))
    # Return the list of sentences 
    return sentences



