import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

credit = pd.read_csv("../input/creditcard.csv")
credit.head()
credit.describe()
credit.info()
credit.shape

credit.plot(figsize=(20, 8))
plt.show()

import seaborn as sns
sns.countplot(x = "Class", data = credit)
credit.loc[:, 'Class'].value_counts()

V_col = credit[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']]
V_col.head()

V_col.hist(figsize=(30, 20))
plt.show()

no_of_normal_transcations = len(credit[credit['Class']==1])
no_of_fraud_transcations = len(credit[credit['Class']==0])
print("no_of_normal_transcations:",no_of_normal_transcations)
print("no_of_fraud_transcations:", no_of_fraud_transcations)

X = credit.iloc[:, 1:29].values
y = credit.iloc[:, 30].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print ("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors = 17)
X,y = credit.loc[:,credit.columns != 'Class'], credit.loc[:,'Class']
knc.fit(X_train,y_train)
y_knc = knc.predict(X_test)
print('accuracy of training set: {:.4f}'.format(knc.score(X_train,y_train)))
print('accuracy of test set: {:.4f}'.format(knc.score(X_test, y_test)))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
print('confusion_matrix of KNN: ', confusion_matrix(y_test, y_knc))
print('precision_score of KNN: ', precision_score(y_test, y_knc))
print('recall_score of KNN: ', recall_score(y_test, y_knc))
print('precision_recall_curve: ', precision_recall_curve(y_test, y_knc))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 20, random_state = 0)
reg.fit(X_train,y_train)
y_rfr = reg.predict(X_test)
reg.score(X_test, y_test)
print('accuracy of training set: {:.4f}'.format(reg.score(X_train,y_train)))
print('accuaracy of test set: {:.4f}'.format(reg.score(X_test, y_test)))

