from kernels import *
import numpy as np
import pandas as pd
from SVM import SVM
from sklearn.metrics import confusion_matrix, accuracy_score
import time

t=time.time()

train0 = pd.read_csv('data/Xtr0_mat100.csv', sep = ' ', header=None).astype(float)
train1 = pd.read_csv('data/Xtr1_mat100.csv', sep = ' ', header=None).astype(float)
train2 = pd.read_csv('data/Xtr2_mat100.csv', sep = ' ', header=None).astype(float)
print(f'length {train0.values.shape} {train1.values.shape} {train2.values.shape}')
X_train = np.vstack((train0.values, train1.values, train2.values))

y_train0 = pd.read_csv('data/Ytr0.csv', sep = ',')
y_train1 = pd.read_csv('data/Ytr1.csv', sep = ',')
y_train2 = pd.read_csv('data/Ytr2.csv', sep = ',')
print(f'length {y_train0.values.shape} {y_train1.values.shape} {y_train2.values.shape}')
y_train = np.vstack((y_train0.values, y_train1.values, y_train2.values))[:,1]
y_train = 2*y_train - 1
print(y_train.shape)

partial = 4000

clf = SVM(PolyKernel(gamma=5, b=1, degree=10), C = 1)
clf.fit(X_train[:partial, :], y_train[:partial])

y_pred_train = clf.predict(X_train[:partial])
print(confusion_matrix(y_train[:partial],y_pred_train))
print(accuracy_score(y_train[:partial],y_pred_train))

X_val = X_train[partial:]
y_pred_val = clf.predict(X_val)
print(confusion_matrix(y_train[partial:], y_pred_val))
print(accuracy_score(y_train[partial:], y_pred_val))

test0 = pd.read_csv('data/Xte0_mat100.csv', sep = ' ', header=None)
test1 = pd.read_csv('data/Xte1_mat100.csv', sep = ' ', header=None)
test2 = pd.read_csv('data/Xte2_mat100.csv', sep = ' ', header=None)
print(f'length {test0.values.shape} {test1.values.shape} {test2.values.shape}')
X_test = np.vstack((test0.values, test1.values, test2.values))
y_predict = clf.predict(X_test)
print(time.time() - t)
y_test = pd.DataFrame(columns=['Id', 'Bound'])
y_test['Bound'] = y_predict
y_test['Bound'].replace(-1, '0', inplace=True)
y_test['Bound'].replace(1, '1', inplace=True)
y_test['Id'] = np.arange(len(y_predict))

y_test.to_csv('test.csv', index=None)