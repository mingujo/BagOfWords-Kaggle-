"""
Purpose:
-----------------------------------------------------------------------------------
- Ensemble Classifier (Gradient Boosting & Logistic Regression)
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup # to extract text only
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer


train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

nltk.download() # find and download 'stopwords' (meaningless words such as 'a', 'is', 'each', 'here')


clean_train_reviews = []

# Loop over each review to put all preprocessed string into the list
for i in xrange( 0, num_reviews ):
	# Print status update
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append(review_to_words(train["review"][i]))