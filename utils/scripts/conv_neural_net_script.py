"""
Purpose:
-----------------------------------------------------------------------------------
- Convolutional Neural Network
- For the model, we actually use published pre-trained vectors trained on part of 
Google News dataset. The model contains 300-dimensional vectors for 3 million words
 and phrases. (zip file is 1.5GB)
-----------------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re


train = pd.read_csv("../../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../data/TestData.tsv", header=0, delimiter="\t", quoting=3)


