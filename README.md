# Kaggle Challenge : Bag of Words Meets Bags of Popcorn
## Min Gu Jo
### Use Google's Word2Vec and TFIDF Vectorizer (Sklearn) for movie reviews


## Overview
Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient. This repo mostly follows the tutorial provided by the competition. 

## Goal
Compare Google's Word2Vec with TFIDF Vectorizer

## Steps
#### Google's word2vec
1) Construct and save a 'Bag of Word' model using 50,000 iMDB movie reviews(.tsv) with Google's word2vec from 'gensim' library
2) Create clusters of semantically related words (bag of centroids) by clustering (K-means clustering)
3) Train a random foerest
 --> Accuracy score of 0.85 when submitted to Kaggle

#### TFIDF Vectorizer
1. Choose the hyper parameters and construct TFIDF vectorizer with both labeled and unlabeled train data set
2. Select the feautres with the top ten percentiles (reduce feature)
3. Train an ensemble (naive bayes & SGDclassifier)


## Directions
1. Clone the repo: `git clone https://github.com/mingujo/BagOfWords-Kaggle-.git'
2. Create 'data' folder, download provided data from https://www.kaggle.com/c/word2vec-nlp-tutorial/data, and unzip
3. Install python dependencies with pip: `pip install -r requirements.txt` 
4. run `utils/scripts/word2vec_model_buildingscript.py` to build 'bag of words' model
5. run `utils/scripts/clustering_script.py` to train, predict, and create a submission file
(run `random_forest_script.py` to train and predict without clustering)

## Accuracy
- TFIDF vectorizer with multinomial naive bayes : 96.180% accuracy (top 60th / top 11% when submitted)
- Word2Vec with K-means clustering and random forest : 85% accuracy (top 300th when submitted)
