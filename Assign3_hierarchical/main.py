import pandas as pd # data processing
from sklearn import preprocessing as preproc
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
import numpy as np



data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]
dataParent = data['parental_level_of_education'].replace('some high school', 'high school')
df = data.loc[data['parental_level_of_education']=='some high school','parental_level_of_education']='high school'
df = pd.DataFrame(data)
dataColmn=['reading score','writing score']





X = data[["reading_score","math_score","writing_score"]].values

Agg_hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
y_hc = Agg_hc.fit_predict(X)
labels = Agg_hc.labels_

cluster_sizes = np.bincount(labels)

for i, size in enumerate(cluster_sizes):
    print(f'Cluster {i}: {size} data points')

Z = linkage(X, 'single')
xx = X.reshape(-1)
def foo(idx):
    return xx[idx]


# Plot the dendrogram
plt.figure()
dendrogram(Z,leaf_label_func=foo, truncate_mode='level', p=5)
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.title("Dendrogram")
plt.show()
"""fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='inferno',alpha=0.7)

ax.set_xlabel("Reading Score")
ax.set_ylabel("Math Score")
ax.set_zlabel("Writing Score")"""

#plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired')


"""X = df[["reading_score","writing_score"]].values
y = df['math_score']

scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(X)
normalized_df = pd.DataFrame(normalize,columns=dataColmn)


pca = PCA(n_components=1)
X = pca.fit_transform(X)
X=pd.DataFrame(X)
xx = pd.concat([X,y],axis=1)

xx.columns = ['PCA', 'math_score']
print(xx)

newdata = xx.values

plt.scatter(newdata[y_hc == 0, 0], newdata[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plotting cluster 2
plt.scatter(newdata[y_hc == 1, 0], newdata[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # plotting cluster 3
plt.scatter(newdata[y_hc == 2, 0], newdata[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plotting cluster 4
plt.scatter(newdata[y_hc == 3, 0], newdata[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  # plotting cluster 5
plt.scatter(newdata[y_hc == 4, 0], newdata[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plot title addition
plt.title('Clusters of customers')
# labelling the x-axis
plt.xlabel('PCA Values')
# label of the y-axis
plt.ylabel('Math scores')
# printing the legend
plt.legend()
# show the plot

"""





"""Z = linkage(X, 'single',)

xx = X.reshape(-1)
print(xx)

plt.figure()
dendrogram(Z,leaf_label_func=foo, truncate_mode='level', p=15)
plt.xlabel("Data Points")

# Label the y-axis
plt.ylabel("Distance")
plt.show()"""

