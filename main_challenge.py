from kernels import *
import numpy as np
import pandas as pd
from SVM import SVM
from sklearn.metrics import confusion_matrix

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

clf = SVM(HadamardKernel(), C = 100)
print(X_train[:20,:].shape)
clf.fit(X_train[:200,:], y_train[:200])

y_pred_train = clf.predict(X_train[:200,:])
print(confusion_matrix(y_train[:200],y_pred_train))


test0 = pd.read_csv('data/Xte0_mat100.csv', sep = ' ', header=None)
test = pd.read_csv('data/Xte1_mat100.csv', sep = ' ', header=None)
test = pd.read_csv('data/Xte2_mat100.csv', sep = ' ', header=None)
print(f'length {test.values.shape} {test.values.shape} {test.values.shape}')
X_train = np.vstack((test.values, test.values, test.values))
#y_predict = clf.predict(X_test)
