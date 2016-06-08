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
			feature_1.append("creditreporting")
		else:
			feature_1.append("ZZZZ")
        
unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]

Y = np.array(numeric_topics)
X = zip(*[feature, question, feature_1])

#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer()
#vectors = vectorizer.fit_transform(question)

#print(vectors)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer =  CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
bag_of_words =vectorizer.fit_transform(X)

clf = MultinomialNB(alpha=.01) 
clf.fit(bag_of_words, Y)

vectors_test_data=[]
vectors_test_expected_result=[]
feature_test = []
feature_1_test = []

with open('test_data.txt') as f:
	for line in f:
		data = line.split(';')
		vectors_test_expected_result.append(data[0])
		vectors_test_data.append(data[1])
		if "mortgage" in data[1]:
			feature_test.append("101")
		elif ("dept" in data[1]) and ("collection" in data[1]):
			feature_test.append("102")
		elif ("credit" in data[1]) and ("reporting" in data[1]):
			feature_test.append("103")
		elif ("student" in data[1]) and ("loan" in data[1]):
			feature_test.append("104")
		elif ("bank" in data[1]) and ("account" in data[1]):
			feature_test.append("105")
		elif ("consumer" in data[1]) and ("loan" in data[1]):
			feature_test.append("106")
		else :
			feature_test.append("107")

		if ("Bank of America" in data[1]):
			feature_1_test.append("creditreporting")
		else:
			feature_1_test.append("ZZZZ")

numeric_result_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in vectors_test_expected_result]
numeric_result_topics = [float(i) for i in numeric_result_topics]

X = zip(*[feature_test, vectors_test_data, feature_1_test])

vectors_test = vectorizer.transform(X)
pred = clf.predict(vectors_test)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != numeric_result_topics[index]):
		numberFalse=numberFalse+1;
	
print ("200 veriden yanlis hesaplanan sayisi:")
print(numberFalse)
print("")
print(pred)

from sklearn.metrics import *
res = confusion_matrix(numeric_result_topics, pred)
print(res)

from sklearn import metrics
acc = metrics.accuracy_score(numeric_result_topics, pred)
print(acc) # 0.793 = (184 + 54)/ (184 + 17 + 45 + 54)

