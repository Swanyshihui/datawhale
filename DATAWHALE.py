

import pandas as pd
from sklearn.metrics import confusion_matrix
train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')
print('Train data shape:',train.shape)
print('TestA data shape:',testA.shape)

from sklearn.metrics import accuracy_score
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
print('ACC:',accuracy_score(y_true, y_pred))

## Precision,Recall,F1-score
 from sklearn import metrics
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
print('Precision',metrics.precision_score(y_true, y_pred))
print('Recall',metrics.recall_score(y_true, y_pred))
print('F1-score:',metrics.f1_score(y_true, y_pred))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings

data_train_sample = pd.read_csv("./train.csv",nrows=5)
chunker = pd.read_csv("./train.csv",chunksize=5) for item in chunker:
    print(type(item))
    #<class 'pandas.core.frame.DataFrame'>
    print(len(item))

missing = data_train.isnull().sum()/len(data_train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()