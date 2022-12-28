
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # figure size
from termcolor import colored as cl # text customization

from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.metrics import explained_variance_score
 # model precision
from sklearn.tree import plot_tree # tree diagram
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]
originalFeatures = data.columns
print('originalFeatures count', len(originalFeatures))
print('originalFeatures', originalFeatures)

dataParent = data['parental_level_of_education'].replace('some high school', 'high school')
df = data.loc[data['parental_level_of_education']=='some high school','parental_level_of_education']='high school'
df = pd.DataFrame(data)
pd.DataFrame({'parental level of education':data.parental_level_of_education})
le = LabelEncoder()
data["gender"] = le.fit_transform(data["gender"])
data["race/ethnicity"] = le.fit_transform(data["race/ethnicity"])
data["test_preparation_course"] = le.fit_transform(data["test_preparation_course"])
data["parental_level_of_education"] = le.fit_transform(data["parental_level_of_education"])
df = pd.DataFrame(data)


X = df[["reading_score","writing_score"]].values
y = df['math_score'].values
print(cl('X variable samples : {}'.format(X[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y[:5]), attrs = ['bold']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
print(cl('X_train shape : {}'.format(X_train.shape), attrs = ['bold'], color = 'black'))
print(cl('X_test shape : {}'.format(X_test.shape), attrs = ['bold'], color = 'black'))
print(cl('y_train shape : {}'.format(y_train.shape), attrs = ['bold'], color = 'black'))
print(cl('y_test shape : {}'.format(y_test.shape), attrs = ['bold'], color = 'black'))

model = DecisionTreeRegressor(criterion='squared_error',max_depth=4)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(explained_variance_score(y_test, predictions))
print(cl('Accuracy of the model is {:.0%}'.format(explained_variance_score(y_test, predictions)), attrs = ['bold']))

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]
originalFeatures = data.columns
df = pd.DataFrame(data)

arr = [6,7]
feature_names = df.columns[arr]
target_names = df["math_score"].unique().tolist()
plot_tree(model, feature_names = feature_names,
         class_names = target_names,
         filled = True,
         rounded = True)

plt.show()





















"""dataParent = data['parental level of education'].replace('some high school', 'high school')
df = data.loc[data['parental level of education']=='some high school','parental level of education']='high school'
df = pd.DataFrame(data)
# x used as reading score
X = df.drop(["math score","writing score","gender","race/ethnicity","parental level of education", "lunch", "test preparation course"], axis=1)
y=df["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)


regr_1 = DecisionTreeRegressor(max_depth=3)
regr_2 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=3", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=4", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()


model = DecisionTreeRegressor(max_depth=4, random_state=44)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print(mean_squared_error(y_test,predictions))
a=model.decision_path(X_train)
print(a)

plt.figure(figsize=(10,8), dpi=300)
plot_tree(model, feature_names=X.columns,class_names=True)
print(explained_variance_score(y_test,predictions))

plt.show()"""