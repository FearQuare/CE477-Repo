from sklearn import preprocessing as preproc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
newData = data.select_dtypes(include=np.number)
dataColmn=['math score','reading score','writing score']
#
scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(newData)
normalized_df = pd.DataFrame(normalize,columns=dataColmn)
print('Original Data \n',newData)
print('Normalized Data by MinMaxScaler() \n',normalized_df)
#
scaler = preproc.StandardScaler()
standardize = scaler.fit_transform(newData)
standardized_df=pd.DataFrame(standardize,columns=dataColmn)
print('Standardized Data by StandardScaler() \n',standardized_df)


