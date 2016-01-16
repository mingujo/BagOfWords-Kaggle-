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

