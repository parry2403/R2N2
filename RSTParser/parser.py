## parser.py
## Author: Yangfeng Ji
## Date: 08-29-2014
## Time-stamp: <yangfeng 10/01/2014 14:56:27>

""" Shift-reduce parser, including following functions
1, Initialize parsing status given a sequence of texts
2, Change the status according to a specific parsing action
3, Get the status of stack/queue
4, Check whether should stop parsing
- YJ
"""

from datastructure import *
from util import *

class SRParser:
    """ It shouldn't be a sub-class of 'object',
        otherwise, it won't work.
        To be clear, being a sub-class of 'object',
        it will make copy of stack and queue, but I
        don't want it works in that way with a purpose.
        - YJ
    """
    def __init__(self, stack, queue):
        """ Initialization
        """
        self.Stack = stack
        self.Queue = queue


    def init(self, texts,pos,dep,lines):
        """ Using text to initialize Queue

        :type texts: list of string
        :param texts: a sequence of EDUs for parsing
        """
    #    print "here"
        for (idx, text) in enumerate(texts):
            n = idx + 1
            node = SpanNode(prop=None)
            node.text = text
            node.eduspan, node.nucspan = (n, n), (n, n)
            node.nucedu = n
            node.pos =pos[text.strip()]
            deps = []
            if text.strip() in  dep:
                dependcies =dep[text.strip()].split("\t")
                for de in dependcies:
                        wrd = de.replace("(","")
                        wrd = wrd.replace(")","")
                        wrd = wrd.replace("'","")
                        if len(wrd)>1:
                               deps.append((wrd.split(",")[0].strip(),wrd.split(",")[1].strip(),wrd.split(",")[2].strip()))




            node.dep = deps
            mlines = []
        #    mlines =lines[text.strip()].split("\t")
         #   for de in mlines:
            mlines.append(lines[text.strip()])
            node.line =mlines
         #   node.dep=dep[text.strip()]
        #    print "here"
        #    print text
        #    print node.dep
            #node.pos =pos
            self.Queue.append(node)


    def operate(self, action_tuple):
        """ According to parsing label to modify the status of
            the Stack/Queue

        Need a special exception for parsing error -YJ

        :type action_tuple: tuple (,,)
        :param action_tuple: one specific parsing action,
                             for example: reduce-NS-elaboration
        """
        action, form, relation = action_tuple
        if action == 'Shift':
            if len(self.Queue) == 0:
                raise ActionError("Shift action with an empty queue")
            node = self.Queue.pop(0)
            self.Stack.append(node)
        elif action == 'Reduce':
            if len(self.Stack) < 2:
                raise ActionError("Reduce action with stack which has less than 2 spans")
            rnode = self.Stack.pop()
            lnode = self.Stack.pop()
            # Create a new node
            # Assign a value to prop, only when it is someone's
            #   children node
            node = SpanNode(prop=None)
            # Children node
            node.lnode, node.rnode = lnode, rnode
            # Parent node of children nodes
            node.lnode.pnode, node.rnode.pnode = node, node
            # Node text
            node.text = lnode.text + " " + rnode.text
            if rnode.pos !=None and lnode.pos !=None :
                   node.pos = lnode.pos + "\t" + rnode.pos

            node.dep=self.getdepinfo(lnode,rnode)
            node.line=self.getlineinfo(lnode,rnode)
        #   return pos
            # EDU span
            node.eduspan = (lnode.eduspan[0],rnode.eduspan[1])
            # Nuc span / Nuc EDU
            if form == 'NN':
                node.nucspan = (lnode.eduspan[0],rnode.eduspan[1])
                node.nucedu = lnode.nucedu
                node.lnode.prop = "Nucleus"
                node.lnode.relation = relation
                node.rnode.prop = "Nucleus"
                node.rnode.relation = relation
            elif form == 'NS':
                node.nucspan = lnode.eduspan
                node.nucedu = lnode.nucedu
                node.lnode.prop = "Nucleus"
                node.lnode.relation = "span"
                node.rnode.prop = "Satellite"
                node.rnode.relation = relation
            elif form == 'SN':
                node.nucspan = rnode.eduspan
                node.nucedu = rnode.nucedu
                node.lnode.prop = "Satellite"
                node.lnode.relation = relation
                node.rnode.prop = "Nucleus"
                node.rnode.relation = "span"
            else:
                raise ValueError("Unrecognized form: {}".format(form))
            self.Stack.append(node)
            # How about prop? How to update it?
        else:
            raise ValueError("Unrecoginized parsing action: {}".format(action))


    def getlineinfo(self,lnode, rnode):
        if lnode.line ==None:
            return rnode.line
        if rnode.line ==None:
            return lnode.line
        newdeps =[]
        for dep in lnode.line:
                newdeps.append(dep)

        for dep in rnode.line:
                newdeps.append(dep)

        #text = lnode.text + " " + rnode.text
        return newdeps
    def getdepinfo(self,lnode, rnode):
        if lnode.dep ==None:
            return rnode.dep
        if rnode.dep ==None:
            return lnode.dep
        newdeps =[]
        for dep in lnode.dep:
        #    print dep
          #  print dep
            text = dep[2]
          #  print text
            if text =='R':
                newdeps.append(dep)
            else:

                if dep[1] in rnode.text:
                    pass
                else:
                    newdeps.append(dep)
        for dep in rnode.dep:
            text = dep[2]
          #  print text
            if text =='R':
                newdeps.append(dep)
            else:

                if dep[1] in lnode.text:
                    pass
                else:
                    newdeps.append(dep)
          #  print text
        #text = lnode.text + " " + rnode.text
        return newdeps

    def getstatus(self):
        """ Return the status of the Queue/Stack
        """
        return (self.Stack, self.Queue)


    def endparsing(self):
        """ Whether we should end parsing
        """
        if (len(self.Stack) == 1) and (len(self.Queue) == 0):
            return True
        elif (len(self.Stack) == 0) and (len(self.Queue) == 0):
            raise ParseError("Illegal stack/queue status")
        else:
            return False

    def getparsetree(self):
        """ Get the entire parsing tree
        """
        if (len(self.Stack) == 1) and (len(self.Queue) == 0):
            return self.Stack[0]
        else:
            return None

            
