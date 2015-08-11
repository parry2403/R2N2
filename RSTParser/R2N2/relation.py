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
from parsers import SRParser
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
rng = numpy.random.RandomState(1234)
from scipy.special import expit
from rst import *
class RSTRelation():
    def __init__(self):
        None
    def checkEDU(self, edu_span, annotations_range,buffer=2):    
        start = int(edu_span[0])+ buffer
        end =int(edu_span[1])-buffer
        for ast,aend in annotations_range:
            if (start in range(int(ast), int(aend))) and (end in range(int(ast), int(aend))): 
    #             print "Yes"
    #             print ast , aend 
    #             print start , end
                return True
        return False   

      
    def addSpan(self,node,edu_span):
        if node.lnode == None and node.lnode ==None:
             edu_span = edu_map[node.text.strip()] 
             edu_span = (edu_span.split("\t")[0],edu_span.split("\t")[1])
             node.trange = edu_span
             return node.trange
        lrange = addSpan(node.lnode,edu_span)
        rrange = addSpan(node.rnode,edu_span)
        node.trange = (lrange[0],rrange[1])
        return node.trange;
     
    def treeTraverseSimple(self,node, height = 0):
        disFeats = []
        if node.lnode != None and node.lnode.prop == 'Nucleus':
    #         print "Nucleus " 
    #         print height
            print node.lnode.trange
        #    text = word_tokenize(node.lnode.text.lower())
        #    for word in text:
        #        nucFeats.append(('Nucleus', word))
           
        elif node.rnode != None and node.rnode.prop == 'Nucleus':
    #         print "Nucleus "
    #         print height
            print node.rnode.trange
            #text = word_tokenize(node.rnode.text.lower())
            #for word in text:
            #    nucFeats.append(('Nucleus', word))
        if None != node.lnode and None != node.rnode :
            treeTraverse(node.lnode,height+1)
            treeTraverse(node.rnode,height+1)
            
    def createHeads(self,node):
            if node !=None:
                hl= self.createHeads(node.lnode)
                hr = self.createHeads(node.rnode)
                node.depList=[]
                if node.lnode == None:
                    node.head = node
                else:
                    if node.lnode.prop == 'Nucleus':
                        node.head = node.lnode.head
                    else:
                        node.head = node.rnode.head
                return max(hl+1,hr+1)
            else: 
                return 0
    def treeTraverseHeads(self,node, height = 0):
        if node!=None:
            print node.head.text   
        if None != node.lnode and None != node.rnode :
            self.treeTraverseHeads(node.lnode,height+1)
            self.treeTraverseHeads(node.rnode,height+1)
         
    def getNucleusParent(self,node,head):
        anc = node.pnode
        while(anc!=None and anc.prop == 'Nucleus'):
            anc = anc.pnode
        anc = anc.pnode
        if anc!=None:
         anc.head.depList.append(node)
        else:
            if head !=node:
                head.depList.append(node)
        
    def getSatelliteParent(self,node):
        anc = node.pnode
        if anc!=None and anc.head!=node:
            anc.head.depList.append(node)
       
    def treeTraverseTerminal(self,node,head, height = 0):
        if None != node.lnode and None != node.rnode :
            self.treeTraverseTerminal(node.lnode,head,height+1)
            self.treeTraverseTerminal(node.rnode,head,height+1)
        else:
            if node.prop == 'Nucleus':
                self.getNucleusParent(node,head)
            else:
                self.getSatelliteParent(node)
                
    def checkEDU(edu_span, annotations_range,annotations_map,primary,height,buffer=2):    
        start = int(edu_span[0])+ buffer
        end =int(edu_span[1])-buffer
        for ast,aend in annotations_range:
            if (start in range(int(ast), int(aend))) and (end in range(int(ast), int(aend))): 
                    if  annotations_map[(ast,aend)] in primary:
                        height_freq[height+1]= height_freq[height+1]+1 
                        node_freq[height+1]= node_freq[height+1]+1
                        return True
                    else:
                        node_freq[height+1]= node_freq[height+1]+1
                        
        return False  

    def treeTraverseDep(self,node,depth_parameter,height=0):   
        local_score = 0.0
        text = word_tokenize(node.text.lower())
        for word in text:
                if word in sa_dict:
    #                print word
                     if height <3:
                        local_score=local_score+sa_dict[word]*(1-depth_parameter*height)
                       
                     #       local_score=local_score+sa_dict[word]*pow(.89, height)
    #                      tcnt[word] = tcnt[word]+ local_score
    #                     data.append(local_score)
                     else:
                         local_score =local_score + .5*sa_dict[word]
    #                 local_score=local_score+ sa_dict[word]*(1-depth_parameter*height)
        score = local_score
    #     /(height+1)
    #     print height , "****---"
    #     print node.text
        
        for nodes in node.depList:
            score = score+ self.treeTraverseDep(self,nodes,depth_parameter,height+1)
        return score
                


    def neval(self,x,alpha = 1e-5):
             x = (2.0/3) * x
             exp2x = np.exp(-2*x)
             val = (1.7159 * (1 - exp2x) / (1 + exp2x)) + (alpha * x.sum())
             return val

    def grad(self,x,alpha = 1e-5):
             val = self.neval(x)
             g =  (1.7159 * (2.0 / 3) * (1 - (val ** 2))) + (np.ones(x.shape) * alpha)
             return g

    def bft(self,root):
            """ Breadth-first traversal on the tree (starting
                from root note!)
            """
            nodelist = []
            if root is not None:
                queue = [root]
                while queue:
                    node = queue.pop(0)
                    if node.lnode is not None:
                        queue.append(node.lnode)
                    if node.rnode is not None:
                        queue.append(node.rnode)
                    nodelist.append(node)
            return nodelist
        
    def is_leaf(self,node):
            """ Whether this node is a leaf node
            """
            if (node.lnode is None) and (node.rnode is None):
                return True
            else:
                return False
    def treeCompositionSimple(self,node,param):
        if node !=None:
                self.treeCompositionSimple(node.lnode,param)
                self.treeCompositionSimple(node.rnode,param)
                comp(node,param)
               
            
    def comp(self,node, param):
        """ Composition function
            For internal node, compute the composition function
            For leaf node, return the value directly

        :type param: instance of Param class or dict of instances
        :param param: composition matrices and bias
        """
     
        imp_rel = ['Constrast','Comparison','antithesis','antithesis-e','consequence-s','concession','Problem-Solution']#'circumstance-e']#'concession']
        tparam = param
        if (node.lnode is None) and (node.rnode is None):
            # If both children nodes are None
            return node.value
        elif (node.lnode is not None) and (node.rnode is not None):
            try:
                # print tparam.S , tparam.N
                if node.rnode.prop=="Nucleus":
                  
                    if node.relation in imp_rel :
                        u = tparam.S.T[1]*(node.lnode.value) + tparam.N.T[1]*(node.rnode.value) 
                    else:
                        u = tparam.S.T[0]*(node.lnode.value) + tparam.N.T[0]*(node.rnode.value) 
#                     u += tparam.N.dot(node.rnode.value) 
                else:
                    if node.relation in imp_rel :
                        u = tparam.S.T[1]*(node.rnode.value) +  tparam.N.T[1]*(node.lnode.value)
                    else:
                        u = tparam.S.T[0]*(node.rnode.value) +  tparam.N.T[0]*(node.lnode.value)
#                     u += tparam.N.dot(node.lnode.value)
            except TypeError:
                print self.lnode.pos, self.lnode.value
                print self.rnode.pos, self.rnode.value
                import sys
                sys.exit()
            node.value = self.neval(u)
        else:
            raise ValueError("Unrecognized situation")   
                
                
    def feedforward(self,nodelist,param,root):
            """ Upward semantic composition step ---
                Using forward to compute the composition result
                of the root node

            Find a topological order of all nodes, then follow
            the linear order to update the node value one by one
            """
           
            # Reverse the nodelist, without change the original one
            nodelist.reverse() # Reverse
            for node in nodelist:
                if self.is_leaf(node): # Leaf node
                    pass
                else: # Non-leaf node
                    self.comp(node,param)
            # Return the composition result
            nodelist.reverse() # Reverse back
            
            return root.value
    def cross_feedforward(self,param,root):
           
                    llist =    self.bft(T.tree.lnode)
                    scorel =  self.feedforward(llist,param, T.tree.lnode)
                    
               
                    rlist =    bft(T.tree.rnode)
                    scorer =  feedforward(rlist,param, T.tree.rnode)
            
                    if T.tree.lnode.prop=="Nucleus":
                        score = scorel*param.N + scorer*param.S
                    else:
                        score = scorel*param.S + scorer*param.N
                    root.value = self.sigmoid(score)
                    return root.value   
    def paramgrad(self,node, param):
        """ Compute the param gradient given current param,
            which is used to update param

        :type param: instance of Param class
        :param param: composition metrices and bias

        :type grad_parent: 1-d numpy.array
        :param grad_parent: gradient information from parent node
        """
       
        # Element-wise multiplication, 1-d numpy.array
        imp_rel = ['Constrast','Comparison','antithesis','antithesis-e','consequence-s','concession','Problem-Solution']
        gu = self.grad(node.value) * node.grad_parent
        if self.is_leaf(node):
            gS = np.zeros(param.S.shape)
            gN = np.zeros(param.N.shape)
        else:
            if node.rnode.prop=="Nucleus":
                if node.relation in imp_rel :
                    gS = np.array([0, np.outer(gu, node.lnode.value)]) 
                    gN = np.array([0, np.outer(gu, node.rnode.value)]) 
                else:
                    gS = np.array([np.outer(gu, node.lnode.value),0]) 
                    gN = np.array([np.outer(gu, node.rnode.value),0]) 
            else:
                if node.relation in imp_rel :
                    gN = np.array([0 , np.outer(gu, node.lnode.value)]) 
                    gS = np.array([0 , np.outer(gu, node.rnode.value)]) 
                else:
                    gN = np.array([np.outer(gu, node.lnode.value),0]) 
                    gS = np.array([np.outer(gu, node.rnode.value),0]) 

        if (node.rnode is None) and (node.lnode is None):
            gbias = np.zeros(param.N.shape)
        else:
            # Without bias term in composition
            gbias = np.zeros(param.N.shape)
        # print gL.shape, gR.shape, gbias.shape
        node.grad_param = Param(gS, gN, gbias)


    def grad_input(self,node, param):
        """ Compute the gradient wrt input from left child node,
            and right child node. If it is a leaf node, return
            the gradient information from parent node directly.
            This is mainly for back-propagating gradient
            information

        :type grad_parent: 1-d numpy.array
        :param grad_parent: gradient information from parent node
        """
        # Compute gradient
        imp_rel = ['Constrast','Comparison','antithesis','antithesis-e','consequence-s','concession','Problem-Solution']
        if (node.lnode is None) and (node.rnode is None):
            try:
                     node.grad = node.grad_parent[0]
            except :
                    node.grad = node.grad_parent
           
            return node.grad_parent
        else:
            # Is it safe to use the value directly?
            # What if it is not updated with the new param?
            
            gu = self.grad(node.value) * node.grad_parent
            # Assign gradient back to children nodes
            # For satellite node
            grad_s = param.S.T[0]* gu
            grad_scont =  param.S.T[1]* gu
            # For nucleus node
            grad_n = param.N.T[0]* gu
            grad_ncont =  param.N.T[1]* gu
            if node.rnode.prop=="Nucleus":
                if node.relation in imp_rel :
                    node.lnode.grad_parent = grad_scont
                    node.rnode.grad_parent = grad_ncont
                else:
                    node.lnode.grad_parent = grad_s
                    node.rnode.grad_parent = grad_n
            else:
                if node.relation in imp_rel :
                    node.rnode.grad_parent = grad_scont
                    node.lnode.grad_parent = grad_ncont
                else:
                    node.rnode.grad_parent = grad_s
                    node.lnode.grad_parent = grad_n
            return (grad_s, grad_n)
        # raise ValueError("Need double check")   

    def ngrad(self,root, param, grad_inputs,nodelist):
        """ Using back-propagation to compute upward gradients
            wrt parameters

        :type grad_input: 1-d numpy.array
        :param grad_input: gradient information from model
        (In other words, this is the gradient of the model
        wrt to the composition result)
        """
        # Assign the grad_input
        root.grad_parent = grad_inputs
        # If the node list is empty, do BFT first
        imp_rel = ['Constrast','Comparison','antithesis','antithesis-e','consequence-s','concession','Problem-Solution']
        # Back propagating the gradient from root to leaf
        for node in nodelist:
            # For model parameters
            self.paramgrad(node,param)
            # For back propagation
            self.grad_input(node,param)
        # Collecting grad information
        # Traversing all the interval nodes and adding
        # all the grad information wrt param together
       
        gparam = Param(np.zeros(nodelist[0].grad_param.S.shape),
                np.zeros(nodelist[0].grad_param.N.shape),
                np.zeros(nodelist[0].grad_param.bias.shape))
        
        gword = {}
        # print '-----------------------------------'
        for node in nodelist:
            # What will happen if the node is leaf node?
                
                if len(node.grad_param.S.shape) ==2:
                    # print node.grad_param.S.shape[1]
                    gs = np.zeros(node.grad_param.S.shape[1], dtype='float')
                    for i in range(node.grad_param.S.shape[1]):
                        gs[i]=node.grad_param.S[0,i] 
                    gn = np.zeros(node.grad_param.N.shape[1], dtype='float')
                    for i in range(node.grad_param.N.shape[1]):
                        gn[i]=node.grad_param.N[0,i]
                else:
                    gparam.S += node.grad_param.S
                    gparam.N += node.grad_param.N
                
                
                if self.is_leaf(node):
                    text = word_tokenize(node.text.lower())
                    for word in text:
                        if isinstance(node.grad, list):
                                val = node.grad[0]
                        else:
                                val = node.grad
                                
                        if word in gword:
                                gword[word] += val
                        else:
                            gword[word] = val
                   
      #  numpy.linalg.norm(gparam.L)
      #  numpy.linalg.norm(gparam.R)
        return gparam , gword
        
    def sigmoid(self,x):
        """ Sigmoid function """
        ###################################################################
        # Compute the sigmoid function for the input here.                #
        ###################################################################
       
        x= 1. / (1. + np.exp(-x))
        return x

    def sigmoid_grad(self,f):
        """ Sigmoid gradient function """
        ###################################################################
        # Compute the gradient for the sigmoid function here. Note that   #
        # for this implementation, the input f should be the sigmoid      #
        # function value of your original input x.                        #
        ###################################################################
        
        
        sig = sigmoid(f)
        return sig * (1 - sig)
       
        
        return f    
    def cross_ngrad(self,root, param, grad_inputs,nodelist):
            """ Using back-propagation to compute upward gradients
                wrt parameters

            :type grad_input: 1-d numpy.array
            :param grad_input: gradient information from model
            (In other words, this is the gradient of the model
            wrt to the composition result)
            """
            # Assign the grad_input
            root.grad_parent = grad_inputs
            # If the node list is empty, do BFT first
           
            # Back propagating the gradient from root to leaf
            for node in nodelist:
                # For model parameters
                if root == node :
                    print"yes"
                    gu = sigmoid_grad(node.value) * node.grad_parent
                
                    if node.rnode.prop=="Nucleus":
                        gS = np.outer(gu, node.lnode.value)
                        gN = np.outer(gu, node.rnode.value)
                    else:
                        gS = np.outer(gu, node.rnode.value)
                        gN = np.outer(gu, node.lnode.value)
                    gbias = np.zeros(param.bias.shape)
                    node.grad_param = Param(gS, gN, gbias)
                    
                    grad_s = np.dot(param.S.T, gu)
                
                # For nucleus node
                    grad_n = np.dot(param.N.T, gu)
                    if node.rnode.prop=="Nucleus":
                        node.lnode.grad_parent = grad_s
                        node.rnode.grad_parent = grad_n
                    else:
                        node.rnode.grad_parent = grad_s
                        node.lnode.grad_parent = grad_n
                else:    
                    paramgrad(node,param)
                    grad_input(node,param)
           
           
            gparam = Param(np.zeros(nodelist[0].grad_param.S.shape),
                    np.zeros(nodelist[0].grad_param.N.shape),
                    np.zeros(nodelist[0].grad_param.bias.shape))
            
            gword = {}
            # print '-----------------------------------'
            for node in nodelist:
                # What will happen if the node is leaf node?
              
                    gparam.S += node.grad_param.S
                    gparam.N += node.grad_param.N
                    
                    
                    if is_leaf(node):
                        text = word_tokenize(node.text.lower())
                        for word in text:
                            if isinstance(node.grad, list):
                                    val = node.grad[0]
                            else:
                                    val = node.grad
                                    
                            if word in gword:
                                    gword[word] += val
                            else:
                                gword[word] = val
                       
          #  numpy.linalg.norm(gparam.L)
          #  numpy.linalg.norm(gparam.R)
            return gparam , gword
        
    def addScoreDep(self,node,depth_parameter,sa_dict,height=0):   
        local_score = 0.0
        text = word_tokenize(node.text.lower())
        for word in text:
                if word in sa_dict:
                         local_score=local_score+sa_dict[word]
        score = local_score

        node.value = np.array([score])
        for nodes in node.depList:
             score=score+self.addScoreDep(nodes,depth_parameter,sa_dict,height+1)
        return score

    def addLearnedScoreDepL(self,node,sa_dict,height=0):   
        local_score = 0.0
        text = node.text
       
    #     score = getEDUData(text,vocab,vocab_no,lr)
        text = word_tokenize(node.text.lower())
        for word in text:
                if word in sa_dict:
                         local_score=local_score+sa_dict[word]
        score = local_score
        node.value = np.array([score])
        for nodes in node.depList:
             score=score+self.addLearnedScoreDepL(nodes,sa_dict,height+1)
        return score

    def learn_word_weights(gword,sa_dict, learning = .05):
          for word in gword:
               
                if word in sa_dict:
                    sa_dict[word] =  sa_dict[word] - learning*gword[word]
    #             else:
    #                  sa_dict[word] =   - learning*gword[word]
    def miniHingeJointTopSGD(pm,files,size,sa_dict,iterations=100):
      alpha  = .05
      learning = 1.0
      i=0
      batches =0
      param = Param(np.ones(1),np.ones(1),np.zeros(1))
    #   param = Param(np.array([.82]),np.array([1.2]),np.zeros(1))
      gparams =[]
      gwords =defaultdict(int)
      for it in range(iterations):
        batches=0
        print "Iteration " , it
        for fname in files:
                T = parse(pm, fname )
                nlist = bft(T.tree)
                max_height =  createHeads( T.tree)
                treeTraverseTerminal(T.tree,T.tree.head)
                depth_parameter = .95/max_height

                addScoreDep(T.tree.head,depth_parameter,sa_dict)

             
                
                if batches< size:
                     batches=batches+1
                else:
                    batches=0
                    N =  sum([gparam.N for gparam in gparams])/size
                    S =  sum([gparam.S for gparam in gparams])/size
                    param.N = param.N - alpha*N
                    param.S = param.S - alpha*S
                    
    #                     else:
    #                         sa_dict[word] =   - learning*gwords[word]/size
                    for word in gwords:
                            if word in sa_dict:
                                sa_dict[word] =  sa_dict[word] - (learning*(gwords[word]) /size)
                        
                    print param.N
                    print param.S
                    if  LA.norm( [param.S,param.N]) >1.8:
                        param.S =(param.S /LA.norm( [param.S,param.N]))
                        param.N =(param.N /LA.norm( [param.S,param.N]))
                    gparams=[]
                    gwords =defaultdict(int)
                    break
                    
                scorel = 0
                scorer= 0 
                if T.tree.lnode !=None and T.tree.lnode !=None:
                   
                    score  = feedforward(nlist,param, T.tree)
                  
                    fname = str(fname.split("/")[-1])
                    rating = int(fname.split(".")[0].split("_")[1])
                    true_pos = 1.6
                    true_neg = -1.6
                    y_i=0
                    if rating <6 :
                        y_i = -1

                    else:
                        y_i=1

                    margin = (1-y_i*score)
                    if margin <0:
                        margin=0
                    if margin==0:
                        grad_inputs=margin     
                    else:
                        grad_inputs=-y_i
                    
                    gpram , gword = ngrad(T.tree, param, grad_inputs,nlist)
                    gparams.append(gpram) 
                    
                    for word in gword:
                        try:
                             val = gword[word]
                             gwords[word] += val
                        except KeyError:
                             gwords[word] = val            
      return param

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

class Param(object):
        def __init__(self, S, N, bias):
            """ Parameters for composition

            :type L: 2-d numpy.array
            :param L: composition matrix for left node

            :type R: 2-d numpy.array
            :param R: composition matrix for right node

            :type bias: 1-d numpy.array
            :param bias: composition bias
            """
            self.S = S
            self.N = N
            self.bias = bias
if __name__ == '__main__':
    
    D =loadmodel("weights.pickle.gz")

    weights = D["words"]
    vocab = D["vocab"]
    vocab_no = D["vocabno"]
    pm = ParsingModel()
    pm.loadmodel("../parsing-model.pickle.gz")
    path = "../../../Movies/edu-input-final/"
    path = "../../../Movies/Bigger-set/"
    files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.edus')]
    # param = miniKJointSGD(files,400,sa_dict,iterations=40)
    param = miniHingeJointTopSGD(pm,files,1500,weights,iterations=100)
    print param.N
    print param.S