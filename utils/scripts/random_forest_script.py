"""
Purpose:
-----------------------------------------------------------------------------------
- Train a random forest classifier with average feature vectors for each review.
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
from clean_data import *
from avgFeatureVec import *
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec #(for loading #Word2Vec object)


############################ Load data and model
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)
model = Word2Vec.load("../../model/300features_40minwords_10context")



############################
# Calculate average feature vectors for training and testing sets,
# using the functions in avgFeatureVec.py. Notice that we now remove stop word.

print "Creating average feature vecs for train reviews"
clean_train_reviews = []
for review in train["review"]:    
    clean_train_reviews.append(review_to_wordlist( review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:    
    clean_test_reviews.append(review_to_wordlist( review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


############################ Random Forest Classifier
# Fit a random forest to the training data, using 100 trees
clf = RandomForestClassifier(n_estimators = 100)

print "Fitting a random forest to labeled training data..."
forest = clf.fit(trainDataVecs, train["sentiment"])
# Test and extract results 
result = forest.predict(testDataVecs)
# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("../../submission/Word2Vec_AverageVectors.csv", index=False, quoting=3)
