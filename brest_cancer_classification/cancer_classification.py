# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer

data =  load_breast_cancer()
# describing the dataset
print(data["DESCR"])
# printing the target set
print(data["target"])
#target names (malignent or benign)
print(data["target_names"])

#feature names
print(data["feature_names"])

#concatenating the target class with the input classes.
dataframe = pd.DataFrame(np.c_[data["data"], data["target"]], columns = np.append(data["feature_names"], ["target"]))

# visualization
sns.pairplot(dataframe, hue = "target" , vars =["mean radius", "mean area", "mean smoothness", "mean texture", "mean perimeter","mean compactness", "mean symmetry"])
sns.scatterplot(x = "mean radius", y = "mean compactness", data = dataframe, hue = "target")
sns.scatterplot(x = "mean radius", y = "mean smoothness", data = dataframe, hue = "target")
sns.scatterplot(x = "mean radius", y = "mean symmetry", data = dataframe, hue = "target")

# checking the correlation between the features.
sns.heatmap(dataframe.corr("kendall"))

# seperating out the input and output for training our model.
X = dataframe.iloc[:, :30 ]
y = dataframe["target"]

# splitting the dataset into training and testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 0)

# Applying standard scaling for a normalization.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying PCA here

from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

varience = pca.explained_variance_ratio_
print(varience)

#checking accuracy with a linear classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state =0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# trying a non-linear classifier
from sklearn.svm import SVC
'''
Hyper_parameters = {'C':[0.1, 1, 10,100], 'gamma':[1,0.1,0.01,0.001],'kernel':['rbf','poly']} 

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),Hyper_parameters,refit = True , verbose = 4 )
grid.fit(X_train, y_train)

print(grid.best_params_)
'''
classifier = SVC(C = 1, gamma = .001,kernel = "rbf")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

## with SVM the accuracy increased to 94.73% and also the classification report is more precise.
