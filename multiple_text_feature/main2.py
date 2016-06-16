import numpy as np
import json
import nltk.stem

from sklearn.naive_bayes import *
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import *

from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO 

from sklearn import tree
X = [[0, 0, 1], [1, 1, 2], [2, 2, 3]]
Y = [0, 1, 2]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

res = clf.predict([[2, 0, 0]])
print(res)

import pydot 
from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=["0", "1","2","3"],  
                         class_names=["0", "1","2"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())