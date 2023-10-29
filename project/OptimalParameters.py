from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def optimal_classifier_parameters(x_train, y_train, X, y, classifier_type):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    acc = []
    if classifier_type == 'MLP':
        for hidden_lay in [(64, 64, 64), (128, 64), (32, 32, 32, 32)]:
            for act in ['logistic', 'tanh', 'relu']:
                for sol in ['adam', 'lbfgs', 'sgd']:
                    indexes = kf.split(x_train, y_train)
                    acc_tmp = []
                    fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
                    for train_index, test_index in indexes:
                        classifier = MLPClassifier(hidden_layer_sizes=hidden_lay, activation=act,
                                                    solver=sol, batch_size=50, learning_rate='constant',
                                                    max_iter=50, random_state=42, early_stopping=True)
                        classifier.fit(x_train.iloc[train_index, :], y_train.iloc[train_index])
                        y_pred = classifier.predict(x_train.iloc[test_index, :])
                        acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
                        fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
                    print(f"For MLP - Hidden layers {hidden_lay}, activation function {act}, and solver {sol}, the accuracy is: {np.mean(acc_tmp)} and confusion matrix is:\n{fin_conf_mat}")
                    acc.append(np.mean(acc_tmp))
    elif classifier_type == 'DT':
        for md in [2, 5, 10, 50, 100]:
            for crt in ['gini', 'entropy']:
                indexes = kf.split(x_train, y_train)
                acc_tmp = []
                fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
                for train_index, test_index in indexes:
                    classifier = DecisionTreeClassifier(max_depth=md, criterion=crt)
                    classifier.fit(x_train.iloc[train_index, :], y_train.iloc[train_index])
                    y_pred = classifier.predict(x_train.iloc[test_index, :])
                    acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
                    fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
                print('For Decision Tree - max_depth=', md, ', criterion=', crt, ' accuracy is: ', np.mean(acc_tmp),
                      ' and confusion matrix is:')
                print(fin_conf_mat)
                acc.append(np.mean(acc_tmp))
    elif classifier_type == 'LR':
        for num in [100, 200, 500, 1000]:
            for solv in ['newton-cg', 'lbfgs', 'sag', 'saga']:
                acc_tmp = []
                fin_conf_mat = np.zeros((len(np.unique(y)), len(np.unique(y))))
                for train_index, test_index in kf.split(x_train, y_train):
                    classifier = LogisticRegression(multi_class='multinomial', max_iter=num, solver=solv)
                    classifier.fit(x_train.iloc[train_index, :], y_train.iloc[train_index])
                    y_pred = classifier.predict(x_train.iloc[test_index, :])
                    acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
                    fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred)
                print('For Logistic Regression - max_iter=', num, ' and solver=', solv, ' accuracy is: ', np.mean(acc_tmp),
                      ' and confusion matrix is:')
                print(fin_conf_mat)
                acc.append(np.mean(acc_tmp))
    
    print('Best accuracy is achieved in iteration number: ', np.argmax(acc))