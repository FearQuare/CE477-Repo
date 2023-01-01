import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing as preproc

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]

dataParent = data['parental_level_of_education'].replace('some high school', 'high school')
df = data.loc[data['parental_level_of_education']=='some high school','parental_level_of_education']
df = pd.DataFrame(data)
pd.DataFrame({'parental level of education':data.parental_level_of_education})


X = df[['reading_score','math_score','writing_score']].values

db = DBSCAN(eps=1, min_samples=4)
# Fit the DBSCAN object to the dataset
db.fit(X)

# Extract the cluster labels for each data point
labels = db.labels_
print(labels)
# Count the number of unique labels
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='Paired',alpha=0.1)

ax.set_xlabel("Reading Score")
ax.set_ylabel("Math Score")
ax.set_zlabel("Writing Score")

#plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Paired')
plt.show()
unique_labels, counts = np.unique(labels, return_counts=True)

# Print the size of each cluster
for label, count in zip(unique_labels, counts):
    print(f'Cluster {label}: {count} data points')


print("Number of clusters:", n_clusters_)


#X = df[['test_preparation_course_completed', 'test_preparation_course_none',
#      'gender_female', 'gender_male', 'race/ethnicity_group A',
#      'race/ethnicity_group B', 'race/ethnicity_group C',
#     'race/ethnicity_group D', 'race/ethnicity_group E',]].values

"""x_col = df[['test_preparation_course_completed', 'test_preparation_course_none',
       'gender_female', 'gender_male', 'race/ethnicity_group A',
       'race/ethnicity_group B', 'race/ethnicity_group C',
       'race/ethnicity_group D', 'race/ethnicity_group E',]].columns


scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(X)
normalized_df = pd.DataFrame(normalize,columns=x_col)
print(normalized_df)

pca = PCA(n_components=3)
X = pca.fit_transform(X)"""

"""colns = ["test_preparation_course","gender","race/ethnicity","parental_level_of_education","lunch"]
df_encoded = pd.get_dummies(df[colns])

df_encoded = pd.concat([df, df_encoded], axis=1)
df_encoded.drop(colns, axis=1, inplace=True)
df = pd.DataFrame(df_encoded)
encoded_columns = df.columns"""
