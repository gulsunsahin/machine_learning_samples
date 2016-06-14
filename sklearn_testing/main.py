import numpy as np
import json
import nltk.stem

from sklearn.naive_bayes import *
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

topic = [] 
question = []
feature = []
feature_1 = []

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
	  def build_analyzer(self):
		  analyzer = super(CountVectorizer, self).build_analyzer()
		  q = lambda doc:doc #(english_stemmer.stem(w) for w in analyzer(doc))
		  return q

with open('data.csv') as f:
	for line in f:
		data = line.split(';')
		topic.append(data[0])
		question.append(data[1])

		if "mortgage" in data[1]:
			feature.append("101")
		elif ("dept" in data[1]) and ("collection" in data[1]):
			feature.append("102")
		elif ("credit" in data[1]) and ("reporting" in data[1]):
			feature.append("103")
		elif ("student" in data[1]) and ("loan" in data[1]):
			feature.append("104")
		elif ("bank" in data[1]) and ("account" in data[1]):
			feature.append("105")
		elif ("consumer" in data[1]) and ("loan" in data[1]):
			feature.append("106")
		else :
			feature.append("107")

		feature_1.append(data[0])
		
X = list(zip(*[question, feature, feature_1]))
X_train, X_test, y_train, y_test = train_test_split(X, topic, test_size=0.30, random_state=20)

vectorizer =  StemmedCountVectorizer(lowercase=False, min_df=2, max_df=0.5, ngram_range = (1,2))
bag_of_words =vectorizer.fit_transform(X_train)

clf = MultinomialNB(alpha=.01) 
clf.fit(bag_of_words, y_train)

vectors_test = vectorizer.transform(X_test)
pred = clf.predict(vectors_test)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != y_test[index]):
		numberFalse=numberFalse+1;
	
print("Yuzde degeri:" + str((numberFalse*100)/len(pred)))
print (str(len(pred)) + " ==> veriden yanlis hesaplanan sayisi:" + str(numberFalse))
