from sklearn.model_selection import train_test_split
from OptimalParameters import optimal_classifier_parameters
from PCA import reduce_dimensionality_pca
from SetUp import load_data, preprocess_data
from Evaluate import evaluate, evaluate_pca


if __name__ == "__main__":
    
    data = load_data()
    X, y = preprocess_data(data)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    optimal_classifier_parameters(x_train, y_train, X, y, classifier_type='MLP')
    optimal_classifier_parameters(x_train, y_train, X, y, classifier_type='DT')
    optimal_classifier_parameters(x_train, y_train, X, y, classifier_type='LR')

    evaluate(x_test, y_test, X, y, 'MLP')
    evaluate(x_test, y_test, X, y, 'DT')
    evaluate(x_test, y_test, X, y, 'LR')
    
    X_pca = reduce_dimensionality_pca(X, y, n_components=0.82)
    x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.05, random_state=42)

    evaluate_pca(x_test, y_test, X_pca, y, 'MLP')
    evaluate_pca(x_test, y_test, X_pca, y, 'DT')
    evaluate_pca(x_test, y_test, X_pca, y, 'LR')