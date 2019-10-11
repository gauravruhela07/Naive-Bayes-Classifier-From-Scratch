def probCol(data,j):
    Yes = 0
    No = 0
    global totalYes, totalNo, X_train
    for i in range(len(X_train)):       
        if X_train[i][j]==data:
            if y_train[i]==2:
                Yes += 1
            else:
                No += 1
    probClassYes = Yes/totalYes
    probClassNo = No/totalNo
    return probClassYes, probClassNo

import pandas as pd

dataset = pd.read_csv('breast-cancer-wisconsin.txt', header=None)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 10].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

totalYes = 0
for i in y_train:
    if i==2:
        totalYes += 1
totalNo = len(y_train) - totalYes

########
#### Code Prediction Section
########

y_pred = []
for j in range(len(X_test)):
    test = X_test[j]
    yes, no = 1, 1
    for i, data in enumerate(test):
        probYes, probNo = probCol(data, i)
        yes = yes * probYes
        no = no * probNo
    yes = yes*(totalYes/len(y_train))
    no = no*(totalNo/len(y_train))
    # print(yes, no)
    try:
        y, n = (yes/(yes+no)), no/(yes+no)
    except:
        y, n = 0.5, 0.5
    if y>=n:
        y_pred.append(2)
    else:
        y_pred.append(4)

import numpy as np
y_pred = np.asarray(y_pred, dtype='int64')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
