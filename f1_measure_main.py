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
def load_txt_pre():
    pre_data=open('E:/tianyidata/preout/pre_1_11_1.txt','r')
    keyline={}
    for one in pre_data.readlines():
        userid=one.strip().split('\t')[0]
        conunt=one.strip().split('\t')[1]
        #print(conunt)
        keyline[userid]=list(eval(conunt))
    #print(keyline)
    return(keyline)

def similarity():
    return 0
def load():
    pre=open('E:/tianyidata/preout/feature_data_line70_1-6.pkl','rb')
    true_7=open('E:/tianyidata/feature_data_line70_7.pkl','rb')
    #mypre=pickle.load(pre)
    mypre=load_txt_pre()
    feature_7=pickle.load(true_7)
    line6={}
    line7={}
    #f-measure
     #R zhaohuilu
    rUserCount=0
    rUserCount=len(feature_7)
    hitUserCount=0
    for one in feature_7.keys():
        if mypre.has_key(one):
            hitUserCount=hitUserCount+1
    print(rUserCount,hitUserCount)

load()