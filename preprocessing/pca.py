import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(x, y, n_components, show_plot=False):
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(x)

    if show_plot:
        colors = 'rgbkcmy'

        unique_y = np.unique(y)
        for i in range(len(unique_y)):
            plt.scatter(pc[y == unique_y[i], 0], pc[y == unique_y[i], 1],
                        color=colors[i % len(colors)],
                        label=unique_y[i])

        plt.legend()
        plt.title('After PCA Transformation')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    print(pca.explained_variance_ratio_, len(pca.explained_variance_))
    print(pca.explained_variance_ratio_.sum())

    columns = []
    for i in range(len(pca.components_)):
        columns.append('principal component ' + str(i + 1))
    pcDf = pd.DataFrame(data=pc, columns=columns)
    finalDf = pd.concat([pcDf, y], axis=1)
    return finalDf


# df = pd.read_csv('data/mushroom/mushrooms-processed.csv')

# y = df.loc[:, 'class']
# x = df.iloc[:, 1:-1]
# print(pca(x, y, 4))
