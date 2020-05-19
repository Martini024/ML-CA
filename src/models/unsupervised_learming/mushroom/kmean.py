import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Part 1
df = pd.read_csv('data/mushroom/mushrooms.csv')
df = df.apply(lambda x: pd.factorize(x)[0])
x = df.loc[:, df.columns != 'class'].values

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(x)

# 2D plot
colors = 'rgbkcmy'

print(clusters.shape)
for i in np.unique(clusters):
    plt.scatter(x[clusters == i, 0], x[clusters == i, 1],
                color=colors[i], label='Cluster ' + str(i + 1))

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='lightskyblue', label='Centroids')
plt.legend()
plt.title('K-Means Clustering')
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.show()


# 3D plot
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    ax.scatter3D(x[clusters == i, 0],
                 x[clusters == i, 1],
                 x[clusters == i, 2],
                 color=colors[i], label='Cluster ' + str(i + 1))

ax.set_xlabel(df.columns[1])
ax.set_ylabel(df.columns[2])
ax.set_zlabel(df.columns[3])

plt.legend()
plt.title('K-Means Clustering')
plt.show()

# Part 2: Find the optimum number of clusters for k-means

wcss = []

# Trying kmeans for k=1 to k=10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()


# Part 3: Actual Categorization
species = np.reshape(df.loc[:, ['Species']].values, (-1,))
i = 0
for label in np.unique(species):
    plt.scatter(x[species == label, 0], x[species == label, 1],
                color=colors[i], label=label)
    i += 1

plt.legend()
plt.title('Species')
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.show()
