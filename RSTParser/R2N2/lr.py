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
from math import pow
from numpy import linalg as LA
from math import fabs
from scipy import sparse
import numpy, sys
import numpy, gzip, sys
from numpy.linalg import norm
from cPickle import load, dump
rng = numpy.random.RandomState(1234)

class LR():
    def __init__(self, alpha,lmbda,maxiter):
        self.alpha = float(alpha) # learning rate for gradient ascent
        self.lmbda = float(lmbda) # regularization constant
        self.epsilon = 0.00001 # convergence measure
        self.maxiter = int(maxiter) # the maximum number of iterations through the data before stopping
        self.threshold = 0.5 # the class prediction threshold

    def getScore(self,fname,theta):
        f = open(fname, 'r')
        content = f.read()
        content = content.decode('utf-8','ignore').encode('ascii','ignore')
        content = content.replace('<br />','').lower()
        words = word_tokenize(content)
        words2 = [w for w in words if not w in stopwords.words('english')]
        score=0
        for word in words2:
            if word in theta:
                # print score
                score= score+theta[word]
        fscore =  1. / (1. + np.exp(-score))
        return fscore

    def cal_grad(self,fname,gwords,grad_input):
        f = open(fname, 'r')
        content = f.read()
        content = content.decode('utf-8','ignore').encode('ascii','ignore')
        content = content.replace('<br />','').lower()
        words = word_tokenize(content)
        words2 = [w for w in words if not w in stopwords.words('english')]
        score=0
        cnt = Counter(words2)
        for word in cnt:
            if word in gwords:
                gwords[word]= gwords[word]+cnt[word]*grad_input
            else:
                gwords[word]=cnt[word]*grad_input
        

    def grad_descent(self,files,theta,vfiles,iterations =500):
        size = len(files)
        
        # print files
        gwords =defaultdict()
        for i in range(iterations):
            print "Iteration " , i
            gwords =defaultdict()
            for f in files:
                # print f
                prob =  self.getScore(f,theta)
                # print score
                # prob = self.sigmoid(score)
                fname = str(f.split("/")[-1])
                rating = int(fname.split(".")[0].split("_")[1])
                y_i=0
                if rating <6 :
                    y_i = 0
                else:
                    y_i=1
                
                grad_input = (prob - y_i)
                self.cal_grad(f,gwords,grad_input)
                
                
            # Update of parameters
            for word in gwords:
                # if word in theta:
                    reg =  (self.lmbda*theta[word])/size
                    theta[word] =  theta[word] - self.alpha*( (gwords[word] /size) +reg)
                # else:
                #      theta[word] = - learning*((gwords[word] /size))
            gwords.clear()
            
            print self.predicts(vfiles,theta)


    def predicts(self,files,theta):
        correct_pred = 0.0
        total_pred = 0.0
        for fname in files:

            score = self.getScore(fname,theta)  
            # print fname  
            fname = str(fname.split("/")[-1])
            # print fname 
            rating = int(fname.split(".")[0].split("_")[1])
            
            if rating <6 and score <0.5:
                correct_pred = correct_pred+1
            if rating >6 and score >0.5:
                correct_pred = correct_pred+1
            total_pred = total_pred+1

        return correct_pred/total_pred
    
    def setup(self,theta,files):
        for fname in files:
            f = open(fname, 'r')
            content = f.read()
            content = content.decode('utf-8','ignore').encode('ascii','ignore')
            content = content.replace('<br />','').lower()
            words = word_tokenize(content)
            words2 = [w for w in words if not w in stopwords.words('english')]
            score=0
            for word in words2:
                if word not in theta:
                    theta[word] = 0 

    def sigmoid(self,x):
        x= 1. / (1. + np.exp(-x))
        return x

    def getDataFiles(self,files,train, vocab , vocab_no):
    
        rows = []
        cols = []
        data = []
        pol = []
        review_sum = count = tp = fp = fn = 0
        folds = ['neg','pos']
       # path = "../../Movies/edu-input-final/"
       
    #     files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.edus')]
        for filename in files:
                # fname = (filename.split(".edus"))[0]
           
                f = open(filename, 'r')
                content = f.read()
                content = content.decode('utf-8','ignore').encode('ascii','ignore')
                content = content.replace('<br />','').lower()
                words = word_tokenize(content)
                words2 = [w for w in words if not w in stopwords.words('english')]
                cnt = Counter(words2)
                for word in cnt:
                    if word not in vocab and train:    
                        vocab[word] = vocab_no[0]
                        vocab_no[0] = vocab_no[0] + 1
                    if word in vocab:
                        rows.append(count)
                        cols.append(vocab[word])
                        data.append(cnt[word])
                

                fname = str(filename.split("/")[-1])
        
                rating = int(fname.split(".")[0].split("_")[1])
                if rating <6:
                    pol.append(-1)
                if rating >6 :
                    pol.append(1)
                    
                count = count + 1
      
        mat = sci.sparse.csr_matrix((data,(rows,cols)), shape=(count,vocab_no[0])).todense()
        return mat, pol


    def fit(self, X, y):
        """
        This function optimizes the parameters for the logistic regression classification model from training 
        data using learning rate alpha and regularization constant lmbda
        @post: parameter(theta) optimized by gradient descent
        """
        # X = self.add_ones(X_) # prepend ones to training set for theta_0 calculations
        
        # initialize optimization arrays
        self.n = X.shape[1] # the number of features
        self.m = X.shape[0] # the number of instances
        self.probas = np.zeros(self.m, dtype='float') # stores probabilities generated by the logistic function
        self.theta = np.zeros(self.n, dtype='float') # stores the model theta

        # iterate through the data at most maxiter times, updating the theta for each feature
        # also stop iterating if error is less than epsilon (convergence tolerance constant)
        print "iter | magnitude of the gradient"
        for iteration in xrange(self.maxiter):
            # calc probabilities
            self.probas = self.get_proba(X)
            print iteration
            # calculate the gradient and update theta
            
            gw = np.zeros(self.n, dtype='float')
            # if y < 0:
            #     y=0
            k = (1.0/self.m) * np.dot(X.T,(self.probas - y).T)
            # print k.shape[0]
            for i in range(k.shape[0]):
                # print k[i,0]
                gw[i]=k[i,0] 
            # g0 = gw[0] # save the theta_0 gradient calc before regularization
           
            gw =gw + (self.lmbda * self.theta)/self.m  # regularize using the lmbda term
            # gw[0] = g0 # restore regularization independent theta_0 gradient calc
          
            self.theta -= self.alpha * gw # update parameters
            
            # calculate the magnitude of the gradient and check for convergence
            loss = np.linalg.norm(gw)
            if self.epsilon > loss:
                break
            
            print iteration, ":", loss

    def get_proba(self, X):
        return 1.0 / (1 + np.exp(- np.dot(X, self.theta)))

    def predict_proba(self, X):
        """
        Returns the set of classification probabilities based on the model theta.
        @parameters: X - array-like of shape = [n_samples, n_features]
                     The input samples.
        @returns: y_pred - list of shape = [n_samples]
                  The probabilities that the class label for each instance is 1 to standard output.
        """
        # X_ = self.add_ones(X)
        return self.get_proba(X)

    def predict(self, X):
        """
        Classifies a set of data instances X based on the set of trained feature theta.
        @parameters: X - array-like of shape = [n_samples, n_features]
                     The input samples.
        @returns: y_pred - list of shape = [n_samples]
                  The predicted class label for each instance.
        """
        y_pred =[]
        pred =  self.predict_proba(X)
       
        for i in range(pred.shape[1]):
            proba =pred[0,i] 
            if proba >self.threshold:
                y_pred.append(1)
            else:
                y_pred.append(-1)
        # y_pred = [proba > self.threshold for proba in self.predict_proba(X)]
        return y_pred

    def add_ones(self, X):
        # prepend a column of 1's to dataset X to enable theta_0 calculations
        # print X.shape[0]
        return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

    def sigmoid_grad(self,f):
        sig = sigmoid(f)
        return sig * (1 - sig)
   
    def compute_accuracy(self,y_test, y_pred):
        """
        @returns: The precision of the classifier, (correct labels / instance count)
        """
        correct = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                correct += 1
        return float(correct) / len(y_test)
    
def savemodel(fname,D):
        """ Save model into fname
        """
        if not fname.endswith('.pickle.gz'):
            fname = fname + '.pickle.gz'
        # D = self.getparams()
        with gzip.open(fname, 'w') as fout:
            dump(D, fout)
        print 'Save model into file {}'.format(fname)


def loadmodel( fname):
        """ Load model from fname
        """
        with gzip.open(fname, 'r') as fin:
            D = load(fin)
        return D
        print 'Load model from file: {}'.format(fname)       

if __name__ == '__main__':
    # path = "../../../Movies//review_polarity/txt_sentoken/pos/"
    # files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    # tfiles = files[0:300] 
    # vfiles = files[950:]
    # path = "../../../Movies//review_polarity/txt_sentoken/neg/"
    # files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    # tfiles = tfiles + files[0:300]
    # vfiles = vfiles + files[950:]
  
    path = "../../../Movies/Bigger-set//"
    tfiles = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    path = "../../../Movies/edu-input-final/"
    vfiles = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.txt')]
    alpha=1.0
    lmbda=0.1 
    maxiter=500
    lr = LR(alpha,lmbda,maxiter)
    theta = defaultdict()
    # print lr.predicts(vfiles,theta)
    # print len(test_pol)
    # print y_pred
    lr.setup(theta,tfiles)
    lr.grad_descent(tfiles,theta,vfiles)
    # print lr.predict(vfiles,theta)
   
