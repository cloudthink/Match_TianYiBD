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

def load():
    feature=open('E:/tianyidata/feature_data_line70.pkl','rb')
    w=open('E:/tianyidata/feature_data_list.txt','w')
    feature=pickle.load(feature)
    featurelist=[]
    for one in feature.keys():
        list=[]
        list=feature[one]
        newlist=[]
        newlist.append([one])
        for i in range(0,70):
            newslist=[]
            for j in range(0,7):
                newslist.append(list[j][i])
            newlist.append(newslist)
        for one in newlist:
            w.write(str(one)+'\t')
        w.write('\n')
        #featurelist.append(newlist)
    #print(featurelist)
    #pickle.dump(featurelist,w)
def yin():
    #print(feature)
    output_6=open('E:/tianyidata/feature_data_line70_1-6.pkl','wb')
    output_7=open('E:/tianyidata/feature_data_line70_7.pkl','wb')

    #pickle.dump(featureline,output)
    line6={}
    line7={}
    for one in feature.keys():
        preline=feature[one]
        line6[one]=preline[0:6]
        line7[one]=preline[6]
    print(line6[one],line7[one])
    #pickle.dump(line6,output_6)
    #pickle.dump(line7,output_7)


load()