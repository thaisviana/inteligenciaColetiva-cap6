import re
import classifier
import naivebayes
import fisherclassifier

def main():
  doc = open("fingerprintGender.txt",'r')
  wordsDict = getwords(doc.read())  
  
  genericClassifier = classifier.classifier(wordsDict)
  genericClassifier.setdb("generic.db")
  sampletrain(genericClassifier)
  
  print "---genericClassifier---"
  print genericClassifier.weightedprob('quick rabbit','good', genericClassifier.fprob)

  print "---Naive Bayes---"
  bayesClassifier = naivebayes.naivebayes(wordsDict)
  bayesClassifier.setdb("bayes.db")
  sampletrain(bayesClassifier)
  print bayesClassifier.prob('quick rabbit','good')
  bayesClassifier.classify('quick money',default='unknown')
  for i in range(10): sampletrain(bayesClassifier)
  print bayesClassifier.classify('quick money',default='unknown')
  
  print "---FISHER CLASSIFIER---"
  fisher = fisherclassifier.fisherclassifier(wordsDict)
  fisher.setdb("fisher.db")
  sampletrain(fisher)
  print fisher.fisherprob('quick rabbit','good')
  print fisher.weightedprob('money','bad', fisher.cprob)


def getwords(doc):
  splitter=re.compile('\\W*')
  # Separa as palavras em caracteres que nao sejam alfabeticos
  words=[s.lower() for s in splitter.split(doc) 
          if len(s)>2 and len(s)<20]
  
 # Retorna um consjunto unico de palavras apenas
  return dict([(w,1) for w in words])

def sampletrain(cl):
  cl.train('Nobody owns the water.','good')
  cl.train('the quick rabbit jumps fences','good')
  cl.train('buy pharmaceuticals now','bad')
  cl.train('make quick money at the online casino','bad')
  cl.train('the quick brown fox jumps','good')


if __name__ == "__main__":
    main()
