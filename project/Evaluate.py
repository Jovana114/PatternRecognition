from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from SetUp import accuracy_per_class, sensitivity_per_class

def evaluate(x_test, y_test, X, y, model_type):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    indexes = kf.split(x_test, y_test)
    fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
    
    for train_index, test_index in indexes:
        if model_type == 'MLP':
            classifier = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                        solver='adam', batch_size=50, learning_rate='constant', 
                                        max_iter=50, random_state=42, early_stopping=True)
        elif model_type == 'DT':
            classifier = DecisionTreeClassifier(max_depth=50, criterion='entropy')
        elif model_type == 'LR':
            classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='newton-cg')
        
        classifier.fit(x_test.iloc[train_index, :].values, y_test.iloc[train_index])
        y_pred = classifier.predict(x_test.iloc[test_index, :].values)
        print(accuracy_score(y_test.iloc[test_index], y_pred))
        fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat, display_labels=classifier.classes_)
    disp.plot(cmap="Blues", values_format='')
    plt.title(model_type)
    plt.show()
    
    print('Percentage of correct predictions: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
    print('Average accuracy: ', accuracy_per_class(fin_conf_mat, y.unique()))
    print('Average sensitivity: ', sensitivity_per_class(fin_conf_mat, y.unique()))

def evaluate_pca(x_test_pca, y_test, X_pca, y, model_type):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    indexes = kf.split(x_test_pca, y_test)
    fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
    
    for train_index, test_index in indexes:
        if model_type == 'MLP':
            classifier = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                        solver='adam', batch_size=50, learning_rate='constant', 
                                        max_iter=50, random_state=42, early_stopping=True)
        elif model_type == 'DT':
            classifier = DecisionTreeClassifier(max_depth=50, criterion='entropy')
        elif model_type == 'LR':
            classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='newton-cg')
        
        classifier.fit(x_test_pca[train_index, :], y_test.iloc[train_index])
        y_pred = classifier.predict(x_test_pca[test_index, :])
        print(accuracy_score(y_test.iloc[test_index], y_pred))
        fin_conf_mat += confusion_matrix(y_test.iloc[test_index], y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat, display_labels=classifier.classes_)
    disp.plot(cmap="Blues", values_format='')
    plt.title(f"{model_type} - PCA")
    plt.show()
    
    print('Percentage of correct predictions - PCA: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
    print('Average accuracy - PCA: ', accuracy_per_class(fin_conf_mat, y.unique()))
    print('Average sensitivity - PCA: ', sensitivity_per_class(fin_conf_mat, y.unique()))
