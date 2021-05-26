import numpy as np
import sklearn
from sklearn import svm
from sklearn.svm import SVC
import pandas as pd

beer_training = pd.read_csv("C:/Users/carlo/OneDrive/Documents/College/4th year/CT4104 Machine Learning/Assigments/A1/training/beer_training.csv", sep="\t")
beer_test = pd.read_csv("C:/Users/carlo/OneDrive/Documents/College/4th year/CT4104 Machine Learning/Assigments/A1/testing/beer_test.csv", sep="\t")

X = beer_training.drop("style", axis=1).values
y = beer_training["style"].values

X_test = beer_test.drop("style", axis=1).values
y_test = beer_test["style"].values

clf = SVC(kernel='linear')
clf.fit(X, y)
prediction = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, prediction)

print("prediction:\n", prediction,"\n")
print("Reality:\n", y_test, "\n")
print("accuracy: ", accuracy)
