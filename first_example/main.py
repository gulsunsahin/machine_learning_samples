import numpy as np
import json
from sklearn.naive_bayes import *

topic = []
question = []


with open('data.csv') as f:
    for line in f:
        data = line.split(',')
        topic.append(data[0])
        question.append(data[1])
        

unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]


Y = np.array(numeric_topics)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(question)

clf = MultinomialNB(alpha=.01) 
clf.fit(vectors, Y)

vectors_test_data=[]
vectors_test_expected_result=[]
with open('test_data.txt') as f:
    for line in f:
        data = line.split(',')
        vectors_test_expected_result.append(data[0])
        vectors_test_data.append(data[1])
        

numeric_result_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in vectors_test_expected_result]
numeric_result_topics = [float(i) for i in numeric_result_topics]

vectors_test = vectorizer.transform(vectors_test_data)
pred = clf.predict(vectors_test)

numberFalse=0
for index in range(len(pred)):
	if (pred[index] != numeric_result_topics[index]):
		numberFalse=numberFalse+1;
	
print ("200 veriden yanlis hesaplanan sayisi:")
print(numberFalse)

vectors_test_data1=["I have to sell my house"]
vectors_test1 = vectorizer.transform(vectors_test_data1)
pred1 = clf.predict(vectors_test1)

print(pred1)




