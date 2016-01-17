"""
Purpose:
-----------------------------------------------------------------------------------
- Create Features from a Bag of Words
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
from sklearn.feature_extraction.text import CountVectorizer #(Sci-Kit Learn's BOW tool)
import nltk.data
from gensim.models import word2vec
from gensim.models import Word2Vec #(for loading #Word2Vec object)

train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)



############################ Load and Preprocess train data


num_reviews = train["review"].size
clean_train_reviews = []

# Loop over each review to put all preprocessed string into the list
for i in xrange( 0, num_reviews ):
	# Print status update
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append(review_to_word(train["review"][i]))

# Feature Extraction. Use the 5000 most frequent words
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,\
                             max_features = 5000) 

# Fit the model and Transform training data into feature vectors
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# train_data_features.shape : 25,000 rows and 5,000 features (one for each vocab)
train_data_features = train_data_features.toarray()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
vocab = vectorizer.get_feature_names()
for tag, count in zip(vocab, dist):
    print count, tag


############################ Random Forest Classifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable

# This may take a few minutes to run
forest = forest.fit(train_data_features, train["sentiment"])


### Fit with Test Set
# Create an empty list and append the clean reviews one by one
num_reviews = test["review"].size # same as training set
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_word(test["review"][i])
    clean_test_reviews.append( clean_review )

# Fit the model and Transform training data into feature vectors
test_data_features = vectorizer.fit_transform(clean_test_reviews)
# train_data_features.shape : 25,000 rows and 5,000 features (one for each vocab)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

############################ Submission file
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv("../../data/Bag_of_Words_model.csv", index=False, quoting=3)
