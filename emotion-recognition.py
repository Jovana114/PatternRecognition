import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Analiza podataka

data = pd.read_csv('emotion_data_gees.csv')
print(data.shape)
print(data.head())

# Proveriti da li ima nedostajućih vrednosti i kako su definisane labele

print('Nedostajuće vrednosti:', data.isnull().sum().sum())

first_column = data.pop('emotion')

data.insert(0, 'emotion', first_column)
print(data.head())

data.drop(['name', 'spk'], axis=1, inplace=True)
print(data.head())

X = data.iloc[:,1:].copy()
y = data.iloc[:,0].copy()
print(X.shape)
print(y.unique())

# Proveriti koliko uzoraka ima u svakoj od klasa, tj da li su jednako zastupljene

print(y.groupby(by=y).count())

# Utvrditi na osnovu statističkih parametara da li postoje razlike između klasa za data obeležja

print(X.groupby(by=y).describe())

# %%
# Definisati funkciju koja na osnovu matrice konfuzije računa tačnost klasifikatora po klasi i prosečnu tačnost.

def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
        print('Za klasu ', klase[i], ' tacnost je: ', tacnost_i[i])
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg


def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg


# %% MLP klasifikator

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = MLPClassifier(hidden_layer_sizes=(64,64,64), activation='tanh',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              learning_rate_init=0.001, max_iter=50, shuffle=True,
                              random_state=42, early_stopping=True, n_iter_no_change=10,
                              validation_fraction=0.1, verbose=False)
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    #y_pred_p = classifier.predict_proba(X.iloc[test_index,:])
    plt.figure
    # iscrtavanje tacnosti val skupa
    plt.plot(classifier.validation_scores_)
    #plt.plot(classifier.loss_curve_)
    plt.show()
    print(accuracy_score(y.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

#print('konacna matrica konfuzije: \n', fin_conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))

# %% KNN - optimal parameters

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
for k in [1, 5, 10]:
    for m in ['euclidean', 'manhattan']:
        indexes = kf.split(X, y)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in indexes:
            classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
            classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
            y_pred = classifier.predict(X.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)
        print('za parametre k=', k, ' i m=', m, ' tacnost je: ', np.mean(acc_tmp), ' a mat. konf. je:')
        print(fin_conf_mat)
        acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

# %% KNN - on data

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    #y_pred_p = classifier.predict_proba(X.iloc[test_index,:])
    plt.figure
    #plt.plot(classifier.loss_curve_)
    plt.show()
    print(accuracy_score(y.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

#print('konacna matrica konfuzije: \n', fin_conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))


# %% Logistic Regression - optimal parameters

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for num in [100, 200, 500, 1000]:
    for solv in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in kf.split(X, y):
            classifier = LogisticRegression(max_iter=num, solver=solv)
            classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
            y_pred = classifier.predict(X.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)
        print('za parametre max_iter=', num, ' i solver=', solv, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
        print(fin_conf_mat)
        acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

# %% Logistic Regression on data

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(X, y)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = LogisticRegression(max_iter=200, solver='liblinear')
    classifier.fit(X.iloc[train_index,:].values, y.iloc[train_index])
    y_pred = classifier.predict(X.iloc[test_index,:].values)
    #y_pred_p = classifier.predict_proba(X.iloc[test_index,:])
    plt.figure
    #plt.plot(classifier.loss_curve_)
    plt.show()
    print(accuracy_score(y.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred)

#print('konacna matrica konfuzije: \n', fin_conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))