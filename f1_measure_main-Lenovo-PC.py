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
def load_txt_pre():
    pre_data=open('E:/tianyidata/preout/pre_1_12_2dayu10.txt','r')
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
    NN=0
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
        for ooone in pre_line:
            ooone=int(ooone)
            pingfang_pre=pingfang_pre+ooone*ooone
        for oooone in true_line:
            #print(oooone)
            oooone=int(oooone)
            pingfang_true=pingfang_true+oooone*oooone
        for ii in range(0,70):
            jisum=jisum+int(pre_line[ii])*int(true_line[ii])
            #pingfang_pre=pingfang_pre+ooone*ooone
            #pingfang_true=pingfang_true+oooone*oooone
        kpingfang_pre=math.sqrt(pingfang_pre)
        kpingfang_true=math.sqrt(pingfang_true)
        if kpingfang_pre==0:
            kpingfang_pre=1.0
            print('1',ooone,oooone)
        if kpingfang_true==0:
            kpingfang_true=1.0
            print('2',ooone,oooone)
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
    print(Similarity,NN,UserCount)
    precisiion=float(Similarity)/float(UserCount)
    recall=float(hitUserCount)/float(rUserCount)
    fmeasure=((2.0)*precisiion*recall)/(precisiion+recall)
    print(precisiion,recall,fmeasure,fmeasure*100.0)




load()