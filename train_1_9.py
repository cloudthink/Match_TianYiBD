__author__ = 'think'

import numpy as np
import urllib
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cross_validation import ShuffleSplit
import numpy as np
import scipy as sp
from collections import Counter

def train():
    train_data=open('E:/tianyidata/train_1_9.txt','r')
    w_data=open('E:/tianyidata/pre_1_10.txt','w')
    data=[]
    for one in train_data.readlines():
        userid=one.strip().split('\t')[0]
        data=one.strip().split('\t')[1]
        print(userid,data,len(data))
train()