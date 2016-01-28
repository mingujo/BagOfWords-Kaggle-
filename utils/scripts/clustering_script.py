"""
Purpose:
-----------------------------------------------------------------------------------
- Perform K-Means Clustering to group the similar sentimental words
- This will lower the noise within random forestc classifier
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
from clean_data import *
from bag_of_centroids import *
from avgFeatureVec import *
from sklearn.cluster import KMeans
import time
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
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, 300)

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:    
    clean_test_reviews.append(review_to_wordlist( review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, 300)



############################ K-means Clustering
start = time.time() 
# Start time. Set "k" (num_clusters) to be 1/5th of the vocabulary size (word vec), 
# i.e. an average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

# Extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
# Build cluster assignment
cluster = kmeans_clustering.fit_predict(word_vectors)
# Get the end time and print how long the process took
end = time.time()
elapsed = end-start
print "Time taken for K Means clustering: ", elapsed, "seconds."

# Create a Word / Index dictionary (cluster)
# Map each vocabulary word to a cluster number
word_centroid_map = dict(zip(model.index2word, cluster))
# For the first 10 clusters
for cluster in xrange(0,10):        
    # Print the cluster number      
    print "\nCluster %d" % cluster        
    # Find all of the words for that cluster number, and print them out    
    words = []    
    for i in xrange(0,len(word_centroid_map.values())):        
        if(word_centroid_map.values()[i] == cluster):            
           words.append(word_centroid_map.keys()[i])    
    print words


############################ Create bags of centroids for our training and test set
# and train a random forest


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 200)
# Fitting the forest may take a few minutes
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

############################ Submission file
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("../../submission/BagOfCentroids.csv", index=False, quoting=3)



