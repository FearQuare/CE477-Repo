import pandas as pd
import numpy as np

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")

math=data['math score']
MQ3, MQ1=np.percentile(math, [75, 25])
IQR = MQ3 - MQ1
m_lowerRange= MQ1 - (1.5 * IQR)
m_upperRange= MQ3 + (1.5 * IQR)
m_low = math[math < m_lowerRange]
m_high = math[math > m_upperRange]
math_low_key=m_low.keys()
math_low_value=m_low.values
math_high_key=m_high.keys()
math_high_value=m_high.values
print("Math Score Outliers:")
for math_low_key, math_low_value in m_low.items():
    print(math_low_key, ":", math_low_value)
for math_high_key, math_high_value in m_high.items():
    print(math_low_key, ":", math_low_value)
print("*************")

reading=data['reading score']
RQ3, RQ1=np.percentile(reading, [75, 25])
IQR = RQ3 - RQ1
r_lowerRange= RQ1 - (1.5 * IQR)
r_upperRange= RQ3 + (1.5 * IQR)
r_low = reading[reading < r_lowerRange]
r_high = reading[reading > r_upperRange]
reading_low_key=r_low.keys()
reading_low_value=r_low.values
reading_high_key=r_high.keys()
reading_high_value=r_high.values

print("Reading Score Outliers:")
for reading_low_key, reading_low_value in r_low.items():
    print(reading_low_key, ":", reading_low_value)
for reading_high_key, reading_high_value in r_high.items():
    print(reading_high_key, ":", reading_high_value)
print("*************")

writing=data['writing score']
WQ3, WQ1=np.percentile(writing, [75, 25])
IQR = WQ3 - WQ1
w_lowerRange= WQ1 - (1.5 * IQR)
w_upperRange= WQ3 + (1.5 * IQR)
w_low = writing[writing < w_lowerRange]
w_high = writing[writing > w_upperRange]
writ_low_key=w_low.keys()
writ_low_value=w_low.values
writ_high_key=w_high.keys()
writ_high_value=w_high.values

print("Writing Score Outliers:")
for writ_low_key, writ_low_value in w_low.items():
    print(writ_low_key, ":", writ_low_value)
for writ_high_key, writ_high_value in w_high.items():
    print(writ_high_key, ":", writ_high_value)
print("*************")