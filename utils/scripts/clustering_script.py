"""
Purpose:
-----------------------------------------------------------------------------------
- K-Means Clustering
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
from clean_data import *
from avgFeatureVec import *
from sklearn.cluster import KMeans
import time
from gensim.models import Word2Vec #(for loading #Word2Vec object)


############################ Load data and model
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)
model = Word2Vec.load("300features_40minwords_10context")


############################
# Calculate average feature vectors for training and testing sets,
# using the functions in avgFeatureVec.py. Notice that we now remove stop word.

print "Creating average feature vecs for train reviews"
clean_train_reviews = []
for review in train["review"]:    
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:    
    clean_test_reviews.append(review_to_wordlist( review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)



############################ K-means Clustering

start = time.time() 
# Start time. Set "k" (num_clusters) to be 1/5th of the vocabulary size (word vec), 
# i.e. an average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)
# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."







