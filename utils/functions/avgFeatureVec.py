"""
Purpose:
-------------------------------------------------------------------------------------
- Take individual word vectors and transform them into a feature set that is the same 
length for every review.
- Since each word is a vector in 300-dimensional space, we can use vector operations 
to combine the words in each review. 
-------------------------------------------------------------------------------------
"""

import numpy as np  

def makeFeatureVec(words, model, num_features):
	"""Average all of the word vectors(feature vectors) in a given paragraph
 
	Parameters
	----------
	words : list of list of strings
		a movie review
	model : Word2Vec object
		Word2Vec model from training data
	num_features : int
		number of features
	
	Returns
	-------
	featureVec : 
		a preprocessed and cleaned review
	"""
	# Initialize an empty numpy array (for speed)    
	featureVec = np.zeros((num_features,), dtype="float32")
	# Initialize a counter (number of words)
	nwords = 0.
	 
	# Index2word is a list that contains the names of the words in the model's vocabulary.  
	index2word_set = set(model.index2word)
	#    
	# Loop over each word in the review and, if it is in the model's vocaublary, add 
	# its feature vector to the total    
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])
	#     
	# Divide the result by the number of words to get the average    
	featureVec = np.divide(featureVec,nwords)
	return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
	""" Given a set of reviews (each one a list of words), calculate the average feature
		vector for each one and return a 2D numpy array
 
	Parameters
	----------
	words : list of list of strings
		a movie review
	model : Word2Vec object
		Word2Vec model from training data
	num_features : int
		number of features
	
	Returns
	-------
	featureVec : 
		a preprocessed and cleaned review

	"""
	# Initialize a counter    
	counter = 0.
	# Preallocate a 2D numpy array, for speed
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
		 
	# Loop through the reviews    
	for review in reviews:
		# Print a status message every 1000th review       
		if counter%1000. == 0.:
			print "Review %d of %d" % (counter, len(reviews))
			#        
			# Call the function (defined above) that makes average feature vectors       
		reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
		# Increment the counter       
		counter = counter + 1.
	return reviewFeatureVecs


