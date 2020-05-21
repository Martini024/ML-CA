import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from processing import preprocessing, pca, feature_selection
from sklearn.metrics import accuracy_score, confusion_matrix


def kmeans(df, y, n_clusters=2):
    df = preprocessing(data=df, y=df[y], perform_scale=True)
    df = feature_selection(df=df, target=df[y], show_process=False)
    # df = pca(df.loc[:, df.columns.difference([y])],
    #          df[y], 0.8, show_result=False)
    x = df.loc[:, df.columns.difference([y])]

    # Applying kmeans to the dataset / Creating the kmeans classifier
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(x.values)
    # print(df[y].values == 0)
    # print(clusters == 0)

    # 2D plot
    colors = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

    plt.subplot(2, 2, 1)
    for i in np.unique(clusters):
        plt.scatter(x.iloc[clusters == i, 0], x.iloc[clusters == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=100, c='lightskyblue', label='Centroids')
    plt.legend()
    plt.title('K-Means Clustering')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    plt.subplot(2, 2, 2)
    for i in np.unique(df[y].values):
        plt.scatter(x.iloc[df[y].values == i, 0], x.iloc[df[y].values == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Ground Truth Classification')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()

    print(accuracy_score(df[y], clusters), confusion_matrix(
        df[y], clusters))

    # # 3D plot
    # fig = plt.figure(figsize=(7, 7))
    # ax = plt.axes(projection='3d')

    # for i in np.unique(clusters):
    #     ax.scatter3D(x[clusters == i, 0],
    #                  x[clusters == i, 1],
    #                  x[clusters == i, 2],
    #                  color=colors[i % 7], label='Cluster ' + str(i + 1))

    # ax.set_xlabel(df.columns[0])
    # ax.set_ylabel(df.columns[1])
    # ax.set_zlabel(df.columns[2])

    # plt.legend()
    # plt.title('K-Means Clustering')
    # plt.show()

    # # Part 2: Find the optimum number of clusters for k-means

    # wcss = []

    # # Trying kmeans for k=1 to k=10
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++')
    #     kmeans.fit(x)
    #     wcss.append(kmeans.inertia_)

    # # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    # plt.plot(range(1, 11), wcss)
    # plt.title('The elbow method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')  # within cluster sum of squares
    # plt.show()


def dbscan(df, y, eps=0.5, min_samples=5):
    df = preprocessing(data=df, y=df[y], perform_scale=True)
    # df = feature_selection(df=df, target=df[y], show_process=False)
    df = pca(df.loc[:, df.columns.difference([y])],
             df[y], 0.8, show_result=True)
    x = df.loc[:, df.columns != y]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(x)

    colors = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

    plt.subplot(2, 2, 1)
    for i in np.unique(clusters):
        label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
        plt.scatter(x.iloc[clusters == i, 0], x.iloc[clusters == i, 1],
                    color=colors[i % 3], label=label)

    plt.legend()
    plt.title('DBSCAN Clustering')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])

    plt.subplots_adjust(wspace=0.4)

    plt.subplot(2, 2, 2)
    for i in np.unique(df[y].values):
        plt.scatter(x.iloc[df[y].values == i, 0], x.iloc[df[y].values == i, 1],
                    color=colors[i % 3], label='Cluster ' + str(i + 1))
    plt.legend()
    plt.title('Ground Truth Classification')
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.show()
