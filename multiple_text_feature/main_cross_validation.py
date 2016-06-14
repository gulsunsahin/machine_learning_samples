import numpy as np
import json
import nltk.stem

from sklearn.naive_bayes import *
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import *
from sklearn import datasets

iris = datasets.load_iris()
n_samples = iris.data.shape[0]

print(n_samples)
print(len(iris.data))

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

		if "financial" in data[1] or "Financial" in data[1]:
			feature_1.append("financialcontains")
		else: 
			feature_1.append("financialnotcontains")

unique_topics = list(set(topic))
new_topic = topic;
numeric_topics = [name.replace('Mortgage', '1').replace('Debt collection', '2').replace('Credit reporting', '3').replace('Consumer Loan', '4').replace('Bank account or service', '5').replace('Money transfers', '6').replace('Credit card', '7').replace('Student loan', '8').replace('Payday loan', '9').replace('Prepaid card', '10').replace('Other financial service', '11') for name in new_topic]
numeric_topics = [float(i) for i in numeric_topics]


Y = np.array(numeric_topics)

#X = list(zip(*[feature_1, question, feature]))
X= question
vectorizer =  StemmedCountVectorizer(lowercase=False, min_df=2, max_df=0.5, ngram_range = (1,2), stop_words='english', max_features=2000)

cv = ShuffleSplit(len(Y), n_iter= 10, test_size = 0.3, random_state=3)

vectorizer =  CountVectorizer(min_df=2, max_df=0.5, stop_words='english', ngram_range = (1,2))
clf = MultinomialNB(alpha=.01) 

for train_index, test_index in cv:
	new_X = []
	new_Y = []
	new_X_Test = []
	new_Y_Test = []
	for index in train_index:
		new_X.append(X[index])
		new_Y.append(Y[index])
	for index in test_index:
		new_X_Test.append(X[index])
		new_Y_Test.append(Y[index])

	bag_of_words =vectorizer.fit_transform(new_X)

	clf.fit(bag_of_words, new_Y)
	vectors_test = vectorizer.transform(new_X_Test)
	pred = clf.predict(vectors_test)
	
	numberFalse=0
	for index in range(len(pred)):
		if (pred[index] != new_Y_Test[index]):
			numberFalse=numberFalse+1;

	print(" ******************************Hesaplanan Deger ***************************************")
	print("Yanlış hesaplama yuzde degeri:" + str((numberFalse*100)/len(pred)))
	print (str(len(pred)) + " ==> veriden yanlis hesaplanan sayisi:" + str(numberFalse))