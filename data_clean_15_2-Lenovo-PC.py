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
    w_data=open('E:/tianyidata/pre_1_20_2.txt','w')
    feature=load_train_data()
    sumline=[]
    cc=[]
    for one in feature.keys():
        #print(feature[one])
        preline=feature[one]
        aprenum=len(preline)
        cc.append(aprenum)
        #print(preline)

        sumday=sumtype=sumcount=0.0
        dayline=[]
        typeline=[]
        for oone in preline:
            #print(oone)
            day=oone[1]
            type=oone[2]
            count=oone[3]
            dayline.append(int(day))
            typeline.append(int(type))
            sumcount=sumcount+float(count)
        #prelingnum=preline.count(0)
        #cc.append(prenum)
        #prenum=aprenum-prelingnum
        #avecount=sumcount/prenum
        #print(aprenum,prelingnum,prenum,sumcount,avecount)
        prefline=[]
        for eday in range(1,8):
            for v in range(1,11):
                if (eday in dayline) and (v in typeline):
                    csum=nmm=0.0
                    for onee in preline:
                        if int(onee[1])==eday or int(onee[2])==v:
                            csum=csum+int(onee[1])+int(onee[2])
                            nmm=nmm+1
                    avecount=csum/(nmm*2)
                    #print(csum,nmm,avecount)
                    prefline.append(abs(int(avecount)))
                    #print(eday,dayline,v,typeline)
                else:
                    #print(eday,dayline,v,typeline)
                    prefline.append(0)
        spre=str(prefline)
        spre=spre[1:-1]
        spre=re.sub(' ','',spre)
        if (aprenum>=5) and (aprenum<=30):
            w_data.write(str(one)+'\t'+spre+'\n')
    print(Counter(cc))
        #print(len(feature[one]))
        #sumline.append(len(feature[one]))
    #print(Counter(sumline))

train()

