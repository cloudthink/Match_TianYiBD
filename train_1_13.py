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

def load_txt_pre():
    pre_data=open('E:/tianyidata/preout/pre_1_11_3.txt','r')
    keyline={}
    for one in pre_data.readlines():
        userid=one.strip().split('\t')[0]
        conunt=one.strip().split('\t')[1]
        #print(conunt)
        keyline[userid]=list(eval(conunt))
    #print(keyline)
    return(keyline)
def train_load():
    time1=time.time()
    train=open('E:/tianyidata/feature_data_line70_1-6.pkl','rb')
    train_feature=pickle.load(train)
    time2=time.time()
    print(time2-time1)
    #print(train_feature)
    return(train_feature)

    #print(mypre.copy())
def train_average():
    train_feature=train_load()
    preline={}
    for one in train_feature.keys():
        #print(one)
        trainline=train_feature[one]
        pline=[]
        for oone in range(0,70):
            sum=average=0.0
            for ooone in range(0,6):
                sum=sum+int(trainline[ooone][oone])
            average=sum/6.0
            if average<1:
                average=1
            pline.append(int(average))
        preline[one]=pline
    #print(preline)
    return(preline)
def train_main():
    train_feature=train_load()
    preline={}
    for one in train_feature.keys():
        #print(one)
        trainline=train_feature[one]
        pline=[]
        for oone in range(0,70):
            sum=average=0.0
            for ooone in range(0,6):
                sum=sum+int(trainline[ooone][oone])
            average=sum/6.0
            if average<1:
                average=1
            pline.append(int(average))
        preline[one]=pline
    #print(preline)
    return(preline)
#train_main()
def load_fm():
    pre=open('E:/tianyidata/preout/feature_data_line70_1-6.pkl','rb')
    true_7=open('E:/tianyidata/feature_data_line70_7.pkl','rb')
    w_data=open('E:/tianyidata/pre_1_13_1.txt','a')
    NN=0
    #mypre=load_txt_pre()
    mypre=train_main()
    #print(mypre)
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
     #P zhunquedu
    UserCount=len(mypre)
    Similarity=precisiion=recall=0.0
    for oone in mypre.keys():
        if feature_7.has_key(oone):
            true_line=feature_7[oone]
            #pre_line=mypre[oone]
        else:
            true_line=[0]*70
        pre_line=mypre[oone]
        sum_fenzi=0.0
        pingfang_pre=pingfang_true=0.0
        jisum=0.0

        for ii in range(0,70):
            pingfang_pre=pingfang_pre+int(pre_line[ii])*int(pre_line[ii])
            pingfang_true=pingfang_true+int(true_line[ii])*int(true_line[ii])
            jisum=jisum+int(pre_line[ii])*int(true_line[ii])
            #pingfang_pre=pingfang_pre+ooone*ooone
            #pingfang_true=pingfang_true+oooone*oooone
        kpingfang_pre=math.sqrt(pingfang_pre)
        kpingfang_true=math.sqrt(pingfang_true)
        if kpingfang_pre==0:
            kpingfang_pre=1.0
            print('1ttttttttt')
        if kpingfang_true==0:
            kpingfang_true=1.0
            print('2222222222222222222222')
        if jisum==0:
            jisum=0.00000000000001
            #print('3',ooone,oooone)
        #print(jisum,kpingfang_true,kpingfang_pre)
        oneSimilarity=jisum/float(kpingfang_pre*kpingfang_true)
        #print(jisum,kpingfang_pre,kpingfang_true,oneSimilarity,pre_line,true_line)
        #print(oneSimilarity)
        Similarity=Similarity+oneSimilarity
        NN=NN+1
        print(NN)
        if oneSimilarity>0.3:
            spre=str(pre_line)
            spre=spre[1:-1]
            spre=re.sub(' ','',spre)
            w_data.write(str(one)+'\t'+spre+'\n')
    print(Similarity,NN,UserCount)
    precisiion=float(Similarity)/float(UserCount)
    recall=float(hitUserCount)/float(rUserCount)
    fmeasure=((2.0)*precisiion*recall)/(precisiion+recall)
    print(precisiion,recall,fmeasure,fmeasure*100.0)


load_fm()