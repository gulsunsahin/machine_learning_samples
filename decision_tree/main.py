import numpy as np
import pandas as pd
import os

#http://hamelg.blogspot.com.tr/2015/11/python-for-data-analysis-part-29.html

titanic_train = pd.read_csv("train.csv")

# Impute median Age for NA Age values
new_age_var = np.where(titanic_train["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       titanic_train["Age"])     # Value if check is false

titanic_train["Age"] = new_age_var 

from sklearn import tree
from sklearn import preprocessing

# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert Sex variable to numeric
encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

# Initialize model
tree_model = tree.DecisionTreeClassifier()

# Train the model
tree_model.fit(X = pd.DataFrame(encoded_sex), 
               y = titanic_train["Survived"])


preds = tree_model.predict_proba(X = pd.DataFrame(encoded_sex))
res = pd.crosstab(preds[:,0], titanic_train["Sex"])

#print(res)

# Make data frame of predictors
predictors = pd.DataFrame([encoded_sex, titanic_train["Pclass"]]).T

# Train the model
tree_model.fit(X = predictors, 
               y = titanic_train["Survived"])

preds = tree_model.predict_proba(X = predictors)

res = pd.crosstab(preds[:,0], columns = [titanic_train["Pclass"], 
                                   titanic_train["Sex"]])

#print(res)

predictors = pd.DataFrame([encoded_sex,
                           titanic_train["Pclass"],
                           titanic_train["Age"],
                           titanic_train["Fare"]]).T

# Initialize model with maximum tree depth set to 8
tree_model = tree.DecisionTreeClassifier(max_depth = 8)

tree_model.fit(X = predictors, 
               y = titanic_train["Survived"])


res = pd.crosstab(preds[:,0], columns = [titanic_train["Age"], titanic_train["Pclass"], 
										 titanic_train["Sex"], 
										 titanic_train["Fare"]])

print(res)

score_res = tree_model.score(X = predictors, 
                 y = titanic_train["Survived"])

print(score_res)