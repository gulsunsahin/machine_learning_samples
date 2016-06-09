import numpy as np
import json
from sklearn.naive_bayes import *
from sklearn import cross_validation

topic = []
question = []
feature = []
feature_1 = []

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

		if ("Bank of America" in data[1]):
			feature_1.append("bankamerica")
		else:
			feature_1.append("otherbank")
        
unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]

Y = np.array(numeric_topics)
X = list(zip(*[feature, question, feature_1]))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, topic, test_size=0.30, random_state=50)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer =  CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
bag_of_words =vectorizer.fit_transform(X_train)

clf = MultinomialNB(alpha=.01) 
clf.fit(bag_of_words, y_train, sample_weight=10)

vectors_test = vectorizer.transform(X_test)
pred = clf.predict(vectors_test)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != y_test[index]):
		numberFalse=numberFalse+1;
	
print("Yuzde degeri:" + str((numberFalse*100)/len(pred)))
print (str(len(pred)) + " ==> veriden yanlis hesaplanan sayisi:" + str(numberFalse))
