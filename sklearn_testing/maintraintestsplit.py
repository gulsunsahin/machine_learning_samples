import numpy as np
import json
import nltk.stem

from sklearn.naive_bayes import *
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

topic = [] 
question = []

with open('data.csv') as f:
	for line in f:
		data = line.split(';')
		topic.append(data[0])
		question.append(data[1])

X_train, X_test, y_train, y_test = train_test_split(question, topic, test_size=0.30, random_state=20)

vectorizer =  CountVectorizer(min_df=2, max_df=0.5, stop_words='english', ngram_range = (1,2))
bag_of_words =vectorizer.fit_transform(X_train)

clf = MultinomialNB(alpha=.01) 
clf.fit(bag_of_words, y_train)

vectors_test = vectorizer.transform(X_test)
pred = clf.predict(vectors_test)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != y_test[index]):
		numberFalse=numberFalse+1;
	
print("Yanlış hesaplama yuzde degeri:" + str((numberFalse*100)/len(pred)))
print (str(len(pred)) + " ==> veriden yanlis hesaplanan sayisi:" + str(numberFalse))
