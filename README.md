# Kaggle Challenge : Bag of Words Meets Bags of Popcorn
## Min Gu Jo
### Use Google's Word2Vec for movie reviews


## Overview
Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient. This repo mostly follows the tutorial provided by the competition. 
(Basic Natural Language Processing & Deep Learning for Text Understanding)


## Steps
1. Train and save a 'Bag of Word' model using 50,000 iMDB movie reviews(.tsv) with Google's word2vec from 'gensim' library
2. Create clusters of semantically related words (bag of centroids) by clustering (K-means clustering)
3. Train a random foerest
 --> Accuracy score of 0.85 when submitted to Kaggle



## Directions
1. Clone the repo: `git clone https://github.com/mingujo/BagOfWords-Kaggle-.git'
2. Install python dependencies with pip: `pip install -r requirements.txt` 
3. run `word2vec_model_buildingscript.py` to build 'bag of words' model
4. run `clustering_script.py` to train, predict, and create a submission file
(run `random_forest_script.py` to train and predict without clustering)

## Issues
- Currently working on CNN(Convolutional Neural Network) script
- Use deep learning library 'keras'