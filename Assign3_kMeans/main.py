import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import pandas as pd # data processing
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]

X = data[["reading_score","math_score","writing_score"]].values
y = data[['writing_score']].values

kmeans = KMeans(3)
kmeans.fit(X)
labels = kmeans.predict(X)

clusterNumber = kmeans.n_clusters
print(clusterNumber)

centroids = kmeans.cluster_centers_
print(centroids)

cluster_sizes = np.bincount(labels)

for i, size in enumerate(cluster_sizes):
    print(f'Cluster {i}: {size} data points')

for i, size in enumerate(centroids):
    print(f'Centroid {i}: {size} ')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points in 3D space
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='Paired', alpha=0.1)

# Overlay the centroids on the scatter plot
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidth=2, c='black')

ax.set_xlabel("Reading Score")
ax.set_ylabel("Math Score")
ax.set_zlabel("Writing Score")

#plt.scatter(X[:, 0], X[:, 1], X[:,2] ,c=labels, cmap='Paired')
#plt.scatter(data['reading_score'],data['math_score'],data['writing_score'])

# Overlay the centroids on the scatter plot
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidth=2, c='black')

#plt.show()


inertias = []

for i in range(1,30):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,30), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


#identified_clusters = kmeans.fit_predict(X)
#prediction = kmeans.predict(y)

#print(identified_clusters)
"""data_with_clusters = data.copy()
data_with_clusters['Clusters'] = clust
#print(data_with_clusters['Clusters'])
plt.plot(data_with_clusters['math_score'])
#plt.scatter(data_with_clusters['math_score'],data_with_clusters['reading_score'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.show()"""

