# -*-coding:utf-8-*-
import locale
locale.setlocale(locale.LC_ALL, '')
# -*- coding: cp1254 -*-

import os
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.tag.stanford import POSTagger
from nltk.parse.stanford import StanfordParser
from multiprocessing import Pool
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
#coding: utf-8
def _parse_output(output_):
        res = []
        cur_lines = []
        for line in output_.splitlines(False):
            if line == '':
            #    res.append(Tree.fromstring('\n'.join(cur_lines)))
                cur_lines = []
            else:
                root =line.split('(')
                if root[0] =='root':
                    return root[1].split(',')[1].split('-')[0].strip()
        return cur_lines
def posgen(fname):
     length = 0
     if fname.endswith('.edus')  :

                            print fname
                            f = open(os.path.join(path,fname),'r')
                            mys1 =os.path.join(path, fname.split(".")[0] +".pos")
                            print mys1
                            pos = open(mys1,"w")
                            data = f.read().splitlines()


                            for line in data:
                                if len(line)>length:
                                    length =len(line)

                                wordb = word_tokenize(line.encode('utf8'))
                                for i in range(len(wordb)):
                                    wordb[i]= wordb[i].encode('utf8')
                                tags = english_postagger.tag(wordb)
                                pos.write(str(line.strip()))
                                pos.write("@#%^&*")
                                # print str(line.strip())
                                # print tags
                                for tgpair in tags[0]:
                                    # print tgpair
                                    pos.write(str(tgpair[1]))
                                    pos.write("\t")
                                pos.write("\n")                               # print i
                               # i=i+1
                               # print length


                      #  continue;
mys = "sentencepos2all"  + ".txt"
#mys1 = "dep2"  + ".txt"
pos = open(mys,"w")
#dep = open(mys1,"w")

english_postagger = POSTagger('../postagger/models/english-bidirectional-distsim.tagger', '../postagger/stanford-postagger.jar', encoding='utf-8')
english_parser = StanfordParser('../postagger/stanford-parser.jar', '../parser/stanford-parser-3.5.0-models.jar', encoding='utf-8')


path = "../../Movies/review_polarity/txt_sentoken/pos/"
p = Pool(25)
p.map(posgen, os.listdir(path))
# for fname in os.listdir(path):
          # posgen(fname)


