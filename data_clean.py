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

def load_train_data():
    train_data=open('E:/tianyidata/part-r-00000.txt','r')
    w_data=open('E:/tianyidata/train_1_10.txt','a')
    featureline={}
    keyfeature={}
    nm=0
    for one in train_data.readlines():
        nm=nm+1
        print(nm)
        danfeature=[]
        linshiline=[]
        lline=[]
        userid=one.strip().split('\t')[0]
        data=one.strip().split('\t')[1]
        type=one.strip().split('\t')[2]
        count=one.strip().split('\t')[3]
        keyfeature[userid]=''
    #for one in keyfeature:
       # w_data.write(one+'\t'+'5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5'+'\n')
#def yin():
        if (nm and userid and data and type and count)!='':
            #danfeature.append(userid)
            danfeature.append(data[1])
            danfeature.append(data[3])
            danfeature.append(type[1:])
            danfeature.append(count)
            #print('dan',danfeature)
        if featureline.has_key(userid) == False:
            linshiline.append(danfeature)
            featureline[userid]=linshiline
            #print(featureline[userid])
        else:
            #print('tt',featureline[userid])
            globline=[]
            oldline=[]
            ooldline=[]
            addline=danfeature
            oldline=featureline[userid]
            ooldline=oldline
            ooldline.append(addline)
            featureline[userid]=ooldline
            #print(ooldline)
            #print(addline,oldline)
            #print('f',addline,oldline,featureline[userid])
    #print(featureline)
    for key in featureline.keys():
        w_data.write(key+'\t'+str(featureline[key])+'\n')
    return(featureline)
            #linshiline.append(lline)
            #keyfeature[userid]=linshiline
        #print(keyfeature)
    #return(featureline)
#def train():

#load_train_data()
def train():
    feature=load_train_data()
    sumline=[]
    for one in feature.keys():
        print(len(feature[one]))
        sumline.append(len(feature[one]))
    print(Counter(sumline))

train()