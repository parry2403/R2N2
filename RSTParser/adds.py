import os
# Read in the file
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
for fname in os.listdir('../aclImdb/test/pos'):
    fullp=os.path.join('../aclImdb/test/pos/', fname)
    fullo=os.path.join('../aclImdbo/test/pos/', fname)
    inf = open(fullp,'r').read()
    inf = inf.decode('ascii', 'ignore')
    s = sent_tokenize(inf)

    out = open(fullo, 'w')
  #  print fname
    for line in s:
          if line[-1]=="." :
               out.write(line +"<s>")
          else:
               out.write(line +"<s>")
  #  out.close

