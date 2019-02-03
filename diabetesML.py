import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#download the dataset from here : https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/diabetes.csv
dataset = pd.read_csv('diabetes.csv')

##Data Visualization##

#Distribution of the dataset over the response variable
import seaborn as sn
sn.countplot(dataset['Outcome'],label="Count")
#check for null values. None in this case
dataset.info()

#Train-Test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'Outcome'], dataset['Outcome'], stratify=dataset['Outcome'])

##Model Fitting##

#k-NN Algorithm
'''
from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
	KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X_train, y_train)
    # record training set accuracy
	training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
	test_accuracy.append(knn.score(X_test, y_test))
'''


#Decision Tree
'''
tree = DecisionTreeClassifier(max_depth=3, random_state=0) #max_depth = 3 gives the highest accuracy
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#Feature Importance in Decision Tree

print("Feature importances:\n{}".format(tree.feature_importances_))

#The DT Algorithm doesn't consider most of the important fields like BMI, Blood Pressure, Age. Hence, it is not an accurate model

'''

#Rnadom Forest Algorithm
'''
rf = RandomForestClassifier(max_depth=3, n_estimators=100) #max_depth = 3 gives the best accuracy and avoids overfitting
rf.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
'''

#MLP with Feature Scaling
#Best model for this dataset
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
