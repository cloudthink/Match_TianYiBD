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
    feature=open('E:/tianyidata/feature_data.pkl','rb')
    feature=pickle.load(feature)
    output=open('E:/tianyidata/feature_data_line70.pkl','wb')
    keyline={}
    for one in feature.keys():
        #print(one,feature[one])
        preline=feature[one]
        prenum=len(preline)
        cc=[]
        cc.append(prenum)
        sumday=sumtype=sumcount=0.0
        chusmalllist=[0]*70
        chubiglist=[chusmalllist,chusmalllist,chusmalllist,chusmalllist,chusmalllist,chusmalllist,chusmalllist]

        for oone in preline:
            #print(oone)
            week=oone[0]
            day=oone[1]
            type=oone[2]
            count=oone[3]
            #print(int(week)-1,int(day)*int(type)-1)
            #print(chubiglist,chusmalllist,len(chubiglist),len(chusmalllist))
            conn=chubiglist[int(week)-1][int(day)*int(type)-1]=int(count)
        #print(chubiglist)
        keyline[one]=chubiglist
    #print(keyline)
    pickle.dump(keyline,output)


load()