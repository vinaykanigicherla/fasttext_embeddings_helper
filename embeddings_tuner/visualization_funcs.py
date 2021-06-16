import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def get_pca_data(data, n_components):
    """Return transformed data and fitted PCA object"""
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca, pca.transform(data)

def plot_all_embeddings(model, is_3d):
    dim = 3 if is_3d else 2
    pca, pca_vectors = get_pca_data(np.array([model.get_word_vector(w) for w in model.words]), dim)
    
    fig = plt.figure(figsize = (8,8))
    if dim > 2:
        ax = fig.add_subplot(1,1,1, projection="3d")
    else:
        ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    
    if dim > 2:
        ax.set_zlabel('Principal Component 3', fontsize = 10)

    ax.set_title(f"{dim}d PCA of FT Vectors", fontsize = 15)

    if dim > 2:
        pca_3d_df = pd.DataFrame(pca_vectors, columns=["pc1", "pc2", "pc3"])
        ax.scatter(pca_3d_df['pc1'], pca_3d_df['pc2'], pca_3d_df['pc3'], s=5)
    else:
        pca_2d_df = pd.DataFrame(pca_vectors, columns=["pc1", "pc2"])
        ax.scatter(pca_2d_df['pc1'], pca_2d_df['pc2'], s=5)
    
    ax.grid()
    plt.show()


def plot_pair_points(pairs, model, is_3d):
    dim = 3 if is_3d else 2
    pca, pca_vectors = get_pca_data(np.array([model.get_word_vector(w) for w in model.words]), dim)   
    
    if is_3d:
        ft_vectors_df = pd.DataFrame(pca_vectors, columns=["pc1", "pc2", "pc3"])
    else:
        ft_vectors_df = pd.DataFrame(pca_vectors, columns=["pc1", "pc2"])
    
    ft_vectors_df["idx"] = model.words
    ft_vectors_df = ft_vectors_df.set_index("idx", drop=True) 

    fig = plt.figure(figsize = (8,8))

    if is_3d:
        ax = fig.add_subplot(1,1,1, projection="3d")
    else:
        ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    
    if is_3d:
        ax.set_zlabel('Principal Component 3', fontsize = 10)

    ax.set_title('FT Vectors of Pairs', fontsize = 15)
    targets = pairs.columns.values
    colors = ['r', 'b']

    for target, color in zip(targets, colors):
        to_plot = ft_vectors_df.loc[pairs[target]]
        if is_3d:
            ax.scatter(to_plot['pc1'], to_plot['pc2'], to_plot['pc3'], c=color, s=5)
        else: 
            ax.scatter(to_plot['pc1'], to_plot['pc2'], c=color, s=5)            
    
    ax.legend(targets)
    ax.grid()
    plt.show()

