__author__ = 'think'
import numpy as np
from numpy import array
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
    output=open('E:/tianyidata/feature_data_line70_0.pkl','wb')
    keyline={}
    for one in feature.keys():
        #print(one,feature[one])
        preline=feature[one]
        prenum=len(preline)
        cc=[]
        cc.append(prenum)
        sumday=sumtype=sumcount=0.0
        chusmalllist0=chusmalllist1=chusmalllist2=chusmalllist3=chusmalllist4=chusmalllist5=chusmalllist6=[0]*70
        chubiglist=array([chusmalllist0,chusmalllist1,chusmalllist2,chusmalllist3,chusmalllist4,chusmalllist5,chusmalllist6])

        for oone in preline:
            #print(oone)
            week=oone[0]
            day=oone[1]
            type=oone[2]
            count=oone[3]
            #print(int(week)-1,int(day)*int(type)-1)
            #print(chubiglist,chusmalllist,len(chubiglist),len(chusmalllist))
            #print('00',chubiglist[int(week)-1][int(day)*int(type)-1])
            #print('11',chubiglist,(int(week)-1),chubiglist[(int(week)-1)])
            chubiglist[(int(week)-1)][(int(day)-1)*10+int(type)-1]=int(count)
            #print('22',chubiglist)
        print('33',one,feature[one],chubiglist)
        keyline[one]=chubiglist
    #print(keyline)
    pickle.dump(keyline,output)
    print('write success')


load()