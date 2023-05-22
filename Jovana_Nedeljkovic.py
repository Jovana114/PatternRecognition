import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# %% Baza podataka

data = pd.read_csv('emotion_data_gees.csv')
print(data.shape)
print(data.head())


# %% Analiza

# Proveriti da li ima nedostajućih vrednosti i kako su definisane labele

print('Nedostajuće vrednosti:', data.isnull().sum().sum())

first_column = data.pop('emotion')

data.insert(0, 'emotion', first_column)
print(data.head())

data.drop(['name'], axis=1, inplace=True)

data['spk'] = data['spk'].replace(['MM', 'MV', 'SK'], 1) # male
data['spk'] = data['spk'].replace(['OK', 'BM', 'SZ'], 2) # female
print(data.head())

label_rows_N = data['emotion'] == 'N'
data = data.drop(data[label_rows_N].index[:12])

label_rows_S = data['emotion'] == 'S'
data = data.drop(data[label_rows_S].index[:12])

label_rows_T = data['emotion'] == 'T'
data = data.drop(data[label_rows_T].index[:12])


X = data.iloc[:,1:].copy()
y = data.iloc[:,0].copy()

print(X.shape)
print(y.unique())

# Proveriti koliko uzoraka ima u svakoj od klasa, tj da li su jednako zastupljene

print(y.groupby(by=y).count())

# Utvrditi na osnovu statističkih parametara da li postoje razlike između klasa za 
# data obeležja

data_an = X.groupby(by=y).describe()
print(X.groupby(by=y).describe())

# Outliers

z_scores = zscore(X)
threshold = 10
outliers = X[(z_scores > threshold).any(axis=1)]
print(outliers)

X = X.drop(outliers.index)
y = y.drop(outliers.index)

print(X.shape)
print(y.unique())

# %% tačnosti i osetljivost 

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
        print('Za klasu ', klase[i], ' tačnost je: ', tacnost_i[i])
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

# %% traint, test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# %% MLP klasifikator - optimal parameters

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_train, y_train)
acc = []
for hidden_lay in [(64,64,64), (128,64), (32,32,32,32)]:
    for act in ['logistic', 'tanh', 'relu']:
        for sol in ['adam', 'lbfgs', 'sgd']:
            indexes = kf.split(x_train, y_train)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
            for train_index, test_index in indexes:
                classifier = MLPClassifier(hidden_layer_sizes=hidden_lay, activation=act,
                                          solver=sol, batch_size=50, learning_rate='constant', 
                                          max_iter=50,random_state=42, early_stopping=True)
                classifier.fit(x_train.iloc[train_index,:], y_train.iloc[train_index])
                y_pred = classifier.predict(x_train.iloc[test_index,:])
                acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
            print(f"For hidden layers {hidden_lay}, activation function {act}, and solver {sol}, the accuracy is: {np.mean(acc_tmp)} and confusion matrix is:\n{fin_conf_mat}")
            print(fin_conf_mat)
            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

# %% MLP klasifikator

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              max_iter=50,random_state=42, early_stopping=True)
    classifier.fit(x_test.iloc[train_index,:].values, y_test.iloc[train_index])
    y_pred = classifier.predict(x_test.iloc[test_index,:].values)
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.title('MLP')
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('prosecna tacnost je: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))


# %% DT klasifikator - optimal parameters

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_train, y_train)
acc = []
for md in [2, 5, 10, 50, 100]:
    for crt in ['gini', 'entropy']:
        indexes = kf.split(x_train, y_train)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in indexes:
            classifier = DecisionTreeClassifier(max_depth=md, criterion=crt)
            classifier.fit(x_train.iloc[train_index,:], y_train.iloc[train_index])
            y_pred = classifier.predict(x_train.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
        print('za parametre max_depth=', md, ', criterion=', crt, ' tacnost je: ', np.mean(acc_tmp),
              ' a mat. konf. je:')
        print(fin_conf_mat)
        acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

# %% DT klasifikator - on data

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = DecisionTreeClassifier(max_depth=50, criterion='entropy')
    classifier.fit(x_test.iloc[train_index,:].values, y_test.iloc[train_index])
    plt.figure(figsize=(16,9), dpi=300)
    tree.plot_tree(classifier)
    y_pred = classifier.predict(x_test.iloc[test_index,:].values)
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='') 
plt.title('DT') 
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('prosecna tacnost je: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))

# %% Logistic Regression - optimal parameters

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for num in [100, 200, 500, 1000]:
    for solv in ['newton-cg', 'lbfgs', 'sag', 'saga']:
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in kf.split(x_train, y_train):
            classifier = LogisticRegression(multi_class='multinomial', max_iter=num, solver=solv)
            classifier.fit(x_train.iloc[train_index,:], y_train.iloc[train_index])
            y_pred = classifier.predict(x_train.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
        print('za parametre max_iter=', num, ' i solver=', solv, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
        print(fin_conf_mat)
        acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

# %% Logistic Regression on data

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', 
                                    solver='newton-cg')
    classifier.fit(x_test.iloc[train_index,:].values, y_test.iloc[train_index])
    y_pred = classifier.predict(x_test.iloc[test_index,:].values)
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.title('MLR') 
plt.show()

print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('prosecna tacnost je: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))


# %% Dimensionality reduction, PCA

# Scaler
s = StandardScaler()
X_std = s.fit_transform(X)
target_names = y.unique()

pca = PCA(n_components=0.82)
pca.fit(X_std)
X_pca = pca.transform(X_std)
print('Redukovani prostor ima dimenziju: ', pca.n_components_)


for i in target_names:
    plt.scatter(X_std[y == i, 0], X_pca[y == i, 1], alpha=.5, label=i)
plt.legend()
plt.title('Dataset')
plt.figure(figsize=(16,9))

for i in target_names:
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=.5, label=i)
plt.legend()
plt.title('PCA on dataset')
plt.figure(figsize=(16,9))

# %%

x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.05, random_state=42)

# %% MLP klasifikator on data - PCA

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu',
                              solver='adam', batch_size=50, learning_rate='constant', 
                              max_iter=50,random_state=42, early_stopping=True)
    classifier.fit(x_test[train_index,:], y_test.iloc[train_index])
    y_pred = classifier.predict(x_test[test_index,:])
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.title('MLP - PCA')
plt.show()

print('procenat tacno predvidjenih - PCA: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('prosecna tacnost je - PCA: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('prosecna osetljivost je - PCA: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))

# %% DT on data - PCA

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = DecisionTreeClassifier(max_depth=50, criterion='entropy')
    classifier.fit(x_test[train_index,:], y_test.iloc[train_index])
    y_pred = classifier.predict(x_test[test_index,:])
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.title('DT - PCA')
plt.show()

print('procenat tacno predvidjenih - PCA: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('Prosečna tačnost je: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('Prosečna osetljivost: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))


# %% Logistic Regression on data - PCA

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
indexes = kf.split(x_test, y_test)
acc = []
fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
for train_index, test_index in indexes:
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='newton-cg')
    classifier.fit(x_test[train_index,:], y_test.iloc[train_index])
    y_pred = classifier.predict(x_test[test_index,:])
    print(accuracy_score(y_test.iloc[test_index], y_pred))
    fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='')  
plt.title('MLR - PCA')
plt.show()

print('procenat tacno predvidjenih - PCA: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
print('Prosečna tačnost je: ', tacnost_po_klasi(fin_conf_mat, y.unique()))
print('Prosečna osetljivost: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))