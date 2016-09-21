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
import re
import cPickle as pickle
import pprint
import math
import time
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
def luojihuigui(x,y):
    model = LogisticRegression()
    model.fit(x, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(x)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
def load_fm():
    feature=open('E:/tianyidata/preout/feature_data_line70_1-6.pkl','rb')
    true_7=open('E:/tianyidata/feature_data_line70_7.pkl','rb')
    w_data=open('E:/tianyidata/pre_1_14_0.txt','a')
    NN=0
    #mypre=load_txt_pre()
    #mypre=train_average()
    #print(mypre)
    feature=pickle.load(feature)
    true=pickle.load(true_7)