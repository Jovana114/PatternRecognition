import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduce_dimensionality_pca(X, y, n_components=0.82):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    target_names = y.unique()

    for i in target_names:
        plt.scatter(X_std[y == i, 0], X_std[y == i, 1], alpha=0.5, label=i)
    plt.legend()
    plt.title('Dataset')
    plt.figure(figsize=(16, 9))

    for i in target_names:
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.5, label=i)
    plt.legend()
    plt.title('PCA on dataset')
    plt.figure(figsize=(16, 9))
    
    return X_pca