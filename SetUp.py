import numpy as np
import pandas as pd
from scipy.stats import zscore

def load_data():
    data = pd.read_csv('emotion_data_gees.csv')
    return data

def preprocess_data(data):
    print('Missing values:', data.isnull().sum().sum())
    
    first_column = data.pop('emotion')
    data.insert(0, 'emotion', first_column)
    data.drop(['name'], axis=1, inplace=True)
    
    data['spk'] = data['spk'].replace(['MM', 'MV', 'SK'], 1)
    data['spk'] = data['spk'].replace(['OK', 'BM', 'SZ'], 2)
    
    label_rows_N = data['emotion'] == 'N'
    data = data.drop(data[label_rows_N].index[:12])
    
    label_rows_S = data['emotion'] == 'S'
    data = data.drop(data[label_rows_S].index[:12])
    
    label_rows_T = data['emotion'] == 'T'
    data = data.drop(data[label_rows_T].index[:12])
    
    X = data.iloc[:, 1:].copy()
    y = data.iloc[:, 0].copy()
    
    z_scores = zscore(X)
    threshold = 10
    outliers = X[(z_scores > threshold).any(axis=1)]
    X = X.drop(outliers.index)
    y = y.drop(outliers.index)
    
    return X, y

def accuracy_per_class(confusion_matrix, classes):
    accuracy_i = []
    N = confusion_matrix.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)), i)
        TP = confusion_matrix[i, i]
        F = 0
        F = (sum(confusion_matrix[i, j]) + sum(confusion_matrix[j, i]))
        TN = np.sum(np.sum(confusion_matrix)) - F - TP
        accuracy_i.append((TP + TN) / np.sum(np.sum(confusion_matrix)))
        print('Za klasu ', classes[i], ' tačnost je: ', accuracy_i[i])
    accuracy_avg = np.mean(accuracy_i)
    return accuracy_avg

def sensitivity_per_class(confusion_matrix, classes):
    sensitivity_i = []
    N = confusion_matrix.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)), i)
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, j])
        sensitivity_i.append(TP / (TP + FN))
        print('Za klasu ', classes[i], ' osetljivost je: ', sensitivity_i[i])
    sensitivity_avg = np.mean(sensitivity_i)
    return sensitivity_avg
