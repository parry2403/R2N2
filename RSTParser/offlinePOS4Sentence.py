__author__ = 'Parry'

import os
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.tag.stanford import POSTagger
from nltk.parse.stanford import StanfordParser
from collections import deque
def _parse_output(sentence,edus,dep):
        res = []
        cur_lines = []
     #   print output_
        wordb = word_tokenize(sentence)
        tags = english_postagger.tag(wordb)
        wrdroot = {}

        depsEDU = []
      #  print curr
        i=0
        curr = edus.popleft()
   #     print curr
        chk = word_tokenize(curr)
     #   print wordb
        for word in wordb:


                     if word not in curr:
                      #   print curr
                     #    print'Split*****',word
                     #    print depsEDU
                         while tags[i][0] != word:
                            depsEDU.append(str(tags[i][1]))
                            i=i+1
                         dep.write(str(curr.strip()))
                         dep.write("@#%^&*")
                         for wrds in depsEDU:
                            dep.write(str(wrds))
                            dep.write("\t")
                         dep.write("\n")
                         depsEDU =[]
                         if len(edus)==0:
                             break
                         curr = edus.popleft()
                         chk = word_tokenize(curr)
                         depsEDU.append(str(tags[i][1]))
                        # while tags[i][0] != word:
                         #                 i=i+1
                         i=i+1
                         depsEDU.append(str(tags[i][1]))
                      #   print curr
                   #  if wordRoot == 'ROOT':
                    #      depsEDU.append((str(word),str(wordRoot),"R"))
                     else:
                      #   print word
                         #  if (str(word),"U") not in depsEDU:
                         try:
                              #  if str(word) not in depsEDU:
                                  #    print word
                               #       while tags[i][0] != word:
                               #           i=i+1
                                      depsEDU.append(str(tags[i][1]))
                                      i=i+1
                         except Exception :
                          #  print edus
                            print 's'
                          #    print depsEDU

        dep.write(str(curr.strip()))
        dep.write("@#%^&*")
        for wrds in depsEDU:
                dep.write(str(wrds))
                dep.write("\t")
        dep.write("\n")
        depsEDU =[]
        return wrdroot
mys = "sentencepos2all"  + ".txt"
#mys1 = "dep2"  + ".txt"
pos = open(mys,"w")
#dep = open(mys1,"w")
english_postagger = POSTagger('../postagger/models/english-bidirectional-distsim.tagger', '../postagger/stanford-postagger.jar')
english_parser = StanfordParser('../postagger/stanford-parser.jar', '../parser/stanford-parser-3.5.0-models.jar')
length = 0
i=0
path = "../data-train/TRAINING/"
for fname in os.listdir(path):

                if fname.endswith('.edus'):
                                print i
                                print fname
                                i=i+1
                                f = open(os.path.join(path,fname),'r')
                                mys1 =os.path.join(path, fname.split(".")[0] +".pos")
                                print mys1


                                dep = open(mys1,"w")
                                data = f.read().splitlines()
                                edus = deque()

                                sentence = None
                                for line in data:
                                       words = word_tokenize(line)
                                       if sentence is None:
                                            sentence = line.strip()
                                        #    print sentence
                                       else:
                                        #   print line.strip()
                                           sentence = sentence + " " + line.strip()
                                       l= words[-1]
                                     #  print 'awesome' in 'wheather is awesome here dude'
                                    #   print l
                        #               print line.strip()
                                       edus.append(line.strip())
                                    #   print line.strip()
                                      # print ".\""
                                   #    print line.strip()[-2:]
                                       if line.strip()[-1]=="." or line.strip()[-2:]==".\"":
                                         #  print sentence
                                        #   print edus
                                           rootWord = _parse_output(sentence,edus,dep)
                                       #    print 'end'
                                           sentence =None
                                           edus = deque()
                                     #  for sentence in sentences:
                                      #    rootWord = _parse_output(english_parser.raw_parse(sentence))
                                     #     dep.write(str(sentence).split())
                                      #    dep.write("@#%^&*")
                                      #    dep.write(str(rootWord))
                                      #    dep.write("\n")
                                      #    print i
                                      #    i=i+1
                                if sentence!=None:
                                     rootWord = _parse_output(sentence,edus,dep)
                                       #    print 'end'
                                     sentence =None
                                     edus = deque()

                                      #  continue;