from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import os
from collections import defaultdict, Counter
from sets import Set
import sklearn
from sklearn import svm
from sklearn import linear_model
import numpy as np
import scipy as sci
import timeit
import sys
sys.path.append('../')
from nltk.stem import WordNetLemmatizer
from buildtree import *
from datastructure import *
from util import *
from model import ParsingModel
from evalparser import *

from math import pow
from numpy import linalg as LA
from math import fabs
from scipy import sparse
import numpy, sys
import numpy, gzip, sys
from numpy.linalg import norm
from util import *
from cPickle import load, dump
from ltan import *
from joint import *
rng = numpy.random.RandomState(1234)

     

if __name__ == '__main__':
   
   
    # path = "../../../Movies/edu-input-final/"
    # vfiles = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.edus')]
   
    alpha=1.0
    lmbda=.1 
    maxiter=100
    rel = .05
    joint =.1
    lr = MaxMargin(alpha,lmbda,maxiter)
   

    D =loadmodel("weights-sd.pickle.gz")
    weights_bow = D["words"]
    vocab = D["vocab"]
    vocab_no = D["vocabno"]

    P =loadmodel("weights-sd.pickle.gz")
    weights_rst = P["words"]
    vocab = D["vocab"]
    vocab_no = D["vocabno"]

    pm = ParsingModel()
    pm.loadmodel("../parsing-model.pickle.gz")
    
    
    # path = "../../../Movies/Bigger-set/"
    # files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.edus')]
  
    path = "../../../Movies//sf/out/"
    files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    tfiles = files
    print len(tfiles)

    # vfiles = files[0:50] + files[950:] 
    path = "../../../Movies//sf/outtest/"
    files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    # tfiles = tfiles + files[50:950]
    vfiles = files[0:5000]
    print len(vfiles)
    ffiles=[]
    # for fname in vfiles:
    #     fname=fname+".line"
    #     print fname
        # f = open(fname, 'r')
        # if fname
    rst =RST()

    i=0
    batches =0
    param = Param(np.ones(1),np.ones(1),np.zeros(1))
    #   param = Param(np.array([.82]),np.array([1.2]),np.zeros(1))
    gparams =[]
    Kb = 1.0
    Kr = 1.0
    size = len(tfiles)
    for it in range(maxiter):
        
        print "Iteration everythng *** " , it

        gwords_bow =defaultdict(int)
        gwords_rst =defaultdict()
        gkbs =[]
        gkrs =[]
        gparams=[]
        
        # correct_pred = 0.0
        # total_pred = 0.0
        # for fname in vfiles:
            
        #     tfile = fname
        #     fname = tfile+".edus"
        #     score_bow =  lr.getScore(tfile,weights_bow)
            
        #     T = parse(pm, fname )
        #     nlist = rst.bft(T.tree)
        #     max_height =  rst.createHeads( T.tree)
        #     rst.treeTraverseTerminal(T.tree,T.tree.head)
        #     depth_parameter = .95/max_height

        #     rst.addLearnedScoreDepL(T.tree.head,weights_rst)
        #     score_rst  = rst.feedforward(nlist,param, T.tree)/1.71
        #     # Combined score
        #     fscore = Kb*score_bow + Kr*score_rst
           
        #     fname = str(fname.split("/")[-1]) 
        #     rating = int(fname.split(".")[0].split("_")[1])
        #     if rating <6 and fscore <0:
        #         correct_pred = correct_pred+1
        #     if rating >6 and fscore >0:
        #         # print "Here " 
        #         correct_pred = correct_pred+1
        #     total_pred = total_pred+1

        # print correct_pred/total_pred

        for fname in tfiles:

            # BOW Grad Start 
            fn = str(fname.split("/")[-1])
            rating = int(fn.split(".")[0].split("_")[1])
            y_i=0
            if rating <6 :
                y_i = -1
            else:
                y_i = 1  

            textfile = fname
            fname = textfile+".edus"
            # textfile = fname.split(".edus")[0]

            score_bow =  lr.getScore(textfile,weights_bow)

            T = parse(pm, fname )
            nlist = rst.bft(T.tree)
            max_height =  rst.createHeads( T.tree)
            rst.treeTraverseTerminal(T.tree,T.tree.head)
            depth_parameter = .95/max_height
            rst.addScoreDep(T.tree.head,depth_parameter,weights_rst)
            score_rst  = rst.feedforward(nlist,param, T.tree)/1.71
            # Combined score
            fscore = Kb*score_bow + Kr*score_rst
            # fscore =score_bow
            margin = (1.0-y_i*fscore)
            if margin <0:
                margin=0

            gradkb = -margin*y_i*score_bow
            gradkr = -margin*y_i*score_rst
            gkbs.append(gradkb)
            gkrs.append(gradkr)

            gradsb = -margin*y_i*Kb
            gradsr = -margin*y_i*Kr
            

            gpram , gword = rst.ngrad(T.tree, param, gradsr,nlist)
            gparams.append(gpram) 
                    
            for word in gword:
                try:
                    val = gword[word]
                    gwords_rst[word] += val
                except KeyError:
                    gwords_rst[word] = val

         
            gscore = lr.tanh_grad(score_bow)
            der = -margin*gscore*y_i*Kb
            lr.cal_grad(textfile,gwords_bow,der)
                                        
              
        # Update Parameters 

            bowweight =  sum([gparam for gparam in gkbs])/size
            rstweight =  sum([gparam for gparam in gkrs])/size
            Kb = Kb - joint*bowweight
            Kr = Kr - joint*rstweight

            # BOW
            for word in gwords_bow:
                    # if word in weights_bow:
                        reg =  (lmbda*weights_bow[word])/size
                        weights_bow[word] =  weights_bow[word] - alpha*( (gwords_bow[word] /size) +reg)
            # RST Update
            N =  sum([gparam.N for gparam in gparams])/size
            S =  sum([gparam.S for gparam in gparams])/size
            param.N = param.N - rel*N
            param.S = param.S - rel*S
            for word in gwords_rst:
                if word in weights_rst:  
                    reg =  (lmbda*weights_rst[word])/size     
                    weights_rst[word] =  weights_rst[word] - alpha*((gwords_rst[word]) /size)
                            
            # print param.N
            # print param.S
            if  LA.norm( [param.S,param.N]) >1.8 or LA.norm( [param.S,param.N]) < -1.8:
                param.S =(param.S /LA.norm( [param.S,param.N]))
                param.N =(param.N /LA.norm( [param.S,param.N]))
            # print param.N
            # print param.S
            # print Kb
            # print Kr
            gwords_bow =defaultdict(int)
            gwords_rst =defaultdict()
            gkbs =[]
            gkrs =[]
            gparams=[]
        # Predictions
        correct_pred = 0.0
        total_pred = 0.0
        for fname in vfiles:
            
            # tfile = (fname.split(".edus"))[0]
            tfile = fname
            fname = tfile+".edus"
            score_bow =  lr.getScore(tfile,weights_bow)
            
            T = parse(pm, fname )
            nlist = rst.bft(T.tree)
            max_height =  rst.createHeads( T.tree)
            rst.treeTraverseTerminal(T.tree,T.tree.head)
            depth_parameter = .95/max_height

            rst.addLearnedScoreDepL(T.tree.head,weights_rst)
            score_rst  = rst.feedforward(nlist,param, T.tree)/1.71
            # Combined score
            fscore = Kb*score_bow + Kr*score_rst
           
            fname = str(fname.split("/")[-1]) 
            rating = int(fname.split(".")[0].split("_")[1])
            if rating <6 and fscore <0:
                correct_pred = correct_pred+1
            if rating >6 and fscore >0:
                # print "Here " 
                correct_pred = correct_pred+1
            total_pred = total_pred+1

        print correct_pred/total_pred
    
    # RST Grad Done 
    # BOW Model Begin
                    
              
    