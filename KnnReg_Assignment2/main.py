import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing as preproc
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score


data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
data.columns = [c.replace(' ','_') for c in data.columns]
originalFeatures = data.columns

dataParent = data['parental_level_of_education'].replace('some high school', 'high school')
df = data.loc[data['parental_level_of_education']=='some high school','parental_level_of_education']='high school'
df = pd.DataFrame(data)
dataColmn=['reading score','writing score']

X = df[["reading_score","writing_score"]].values
y = df['math_score'].values

#scaler = preproc.MinMaxScaler()
#normalize = scaler.fit_transform(X)
#normalized_df = pd.DataFrame(normalize,columns=dataColmn)
#pca = PCA(n_components=1)
#X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
print(X_train.size)
print(y_train.size)

rmse_val = [] #to store rmse values for different k
for K in range(50):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test)#make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred) )#calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
      # elbow curve

curve = pd.DataFrame(rmse_val)
plt.plot(curve)
plt.title('Elbow Curve')
plt.xlabel('k-value')
plt.ylabel('Error (RMSE)')
plt.show()

k_range = range(1, 26)
testing_scores = []
training_scores = []
for k in k_range:
    knn = neighbors.KNeighborsRegressor(n_neighbors = k, weights='uniform', algorithm='auto')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_test = y_test.reshape(-1,1)
    testing_scores.append(knn.score(X_train, y_train))
    training_scores.append(knn.score(X_test, y_test))

print(testing_scores)
plt.plot(k_range, testing_scores, label='testing score')
plt.plot(k_range,training_scores, label='training score')
plt.legend()
plt.xlabel('Value of K for KNN')
plt.ylabel('R-square Score')
print('Explained Variance Score: ', explained_variance_score(y_test,y_pred))
plt.show()

