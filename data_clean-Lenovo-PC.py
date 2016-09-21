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

def load_train_data():
    train_data=open('E:/tianyidata/part-r-00000_2.txt','r')

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
        if (nm and userid and data and type and count)!='' and data[1]!=7:
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
    #for key in featureline.keys():
     #   w_data.write(key+'\t'+str(featureline[key])+'\n')
    return(featureline)
            #linshiline.append(lline)
            #keyfeature[userid]=linshiline
        #print(keyfeature)
    #return(featureline)
#def train():

#load_train_data()
def train():
    w_data=open('E:/tianyidata/pre_1_12_3ddy1.txt','a')
    feature=load_train_data()
    sumline=[]
    cc=[]
    for one in feature.keys():
        #print(feature[one])
        preline=feature[one]
        prenum=len(preline)

        cc.append(prenum)
        sumday=sumtype=sumcount=0.0
        for oone in preline:
            #print(oone)
            day=oone[1]
            type=oone[2]
            count=oone[3]
            sumday=sumday+float(day)
            sumtype=sumtype+float(type)
            sumcount=sumcount+float(count)
        aveday=sumday/prenum
        avetype=sumtype/prenum
        avecount=sumcount/prenum
        print('key',avecount,aveday,avetype,prenum)
        prefline=[]
        for eday in range(1,8):
            for v in range(1,11):
                #print(eday,v)
                chaday=abs(eday-aveday)
                chatype=abs(v-avetype)
                precount=(avecount*(1.0-chaday/aveday)+(1.0-chatype/avetype)*avecount)*0.5
                prefline.append(int(precount))
        spre=str(prefline)
        spre=spre[1:-1]
        spre=re.sub(' ','',spre)
        if prenum>=1:
            w_data.write(str(one)+'\t'+spre+'\n')
    print(Counter(cc))
        #print(len(feature[one]))
        #sumline.append(len(feature[one]))
    #print(Counter(sumline))

train()

