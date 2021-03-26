# -*- coding: utf-8 -*-


from SVM import SVM
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from kernels import MismatchKernel
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils import normalize_K
from utils import create_submission

files = {(12, 1): 'kernels/kernel_k_12_1.p',
         (12, 2): 'kernels/kernel_k_12_2.p',
         (13, 3): 'kernels/kernel_k_13_3.p',
         (14, 2): 'kernels/kernel_k_14_2.p',
         (11, 1): 'kernels/kernel_k_11_1.p',
         (15, 2): 'kernels/kernel_k_15_2.p',
         (14, 1): 'kernels/kernel_k_14_1.p',
         (13, 0): 'kernels/kernel_k_13_0.p',
         (13, 2): 'kernels/kernel_k_13_2.p',
         (14, 0): 'kernels/kernel_k_14_0.p',
         (17, 2): 'kernels/kernel_k_17_2.p',
         (14, 3): 'kernels/kernel_k_14_3.p'}

dataframes_test = []
labels = []

trainings = [range(2000), range(2000,4000), range(4000,6000)]
testings = [range(6000,7000), range(7000,8000), range(8000,9000)]

for i in range(3):
    dataframes_test.append(pd.read_csv(f'data/Xte{i}.csv').values[:,1:])
    labels.append(2*pd.read_csv(f'data/Ytr{i}.csv')['Bound'].values-1)

kernel_names = [ [(13,0),(14,0),(14,2),(17,2)],
         [(12,1), (14,1), (14,3), (17,2)],
         [(11,1),(13,2),(12,2),(13,3),(14,3)]]

solutions = []
for i,idx in enumerate(kernel_names):
    K = np.zeros((9000,9000))
    
    training = trainings[i]
    testing = testings[i] 
    
    for kernel_name in kernel_names[i]:
        file_name = files[kernel_name]
        with open(file_name, 'rb') as f:
            temp_k = pickle.load(f)
            K += normalize_K( temp_k.astype(np.float64) )

    
    clf = SVM(C = 1e-5, kernel=MismatchKernel)
    clf.fit(y=labels[i], K = K[training][:, training])
    solutions.append(clf.predict(pairwise_K=K[testing][:,training]))
    
solutions = np.concatenate(solutions)
submission_file_name = "Yte.csv"
create_submission(solutions, submission_file_name)




    

