"""
Purpose:
-----------------------------------------------------------------------------------
- Preprocess the review data
- Build a model
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
from clean_data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectPercentile, f_classif

############################ Load data
train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)
# unlabled_train has additional 50000 reviews
unlabeled_train = pd.read_csv( "../../data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )


############################ Parse data
print "Creating a list of review paragraphs for train reviews..."
clean_train_reviews = []
for review in train["review"]:    
    clean_train_reviews.append(" ".join(review_to_wordlist(review, remove_stopwords=True)))

clean_unlabeled_train_reviews = []
for review in unlabeled_train["review"]:    
    clean_unlabeled_train_reviews.append(" ".join(review_to_wordlist(review, remove_stopwords=True)))


print "Creating a wordlist for test reviews..."
clean_test_reviews = []
for review in test["review"]:    
    clean_test_reviews.append(" ".join(review_to_wordlist( review, remove_stopwords=True)))


############################ Create a vectorizer (TFIDF vectorizer)
print "Vectorizing with TFIDF..."
# min_df = 2 (document frequency boundary)
# max_df = 0.95 ()
# max_features = 200000 (number of top 200000 max_features ordered by term frequency)
# ngram_range = (1,4)   (look up from 1 to 4 near words)
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, max_features = 200000, ngram_range = (1, 4),
                              sublinear_tf = True )

vectorizer = vectorizer.fit(clean_train_reviews + clean_unlabeled_train_reviews)
train_data_features = vectorizer.transform(clean_train_reviews)
test_data_features = vectorizer.transform(clean_test_reviews)

############################ Reduce feature
# we just choose the top 10 percentile 
# decreasing percentitle increases accuracy upto some point
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(train_data_features, train["sentiment"])
reduced_train_data_features = selector.transform(train_data_features).toarray()
reduced_test_data_features = selector.transform(test_data_features).toarray()



############################ Train an ensemble model
# classification with discrete features (fits well with tfidf) (word counts for text classification)
model_MNB = MultinomialNB(alpha=0.00005)
model_MNB.fit(reduced_train_data_features, train["sentiment"])
model_SGD = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model_SGD.fit(reduced_train_data_features, train["sentiment"] )


############################ Predict label
p1 = model_MNB.predict_proba(reduced_test_data_features)[:,1]
p2 = model_SGD.predict_proba(reduced_test_data_features)[:,1]

############################ Make a submission
print "Making a submission file..."

output = pd.DataFrame( data = { "id": test["id"], "sentiment": .2*p1 + 1.*p2 } )
output.to_csv('../../submission/tfidf_final.csv', index = False, quoting = 3 )


