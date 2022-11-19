import pandas as pd
import numpy as np

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")

print(data.isna())
print(data.isnull())
print(data == float('nan'))
print(data.isnull().any())
#do not have any NA, null or NaN values in dataset


