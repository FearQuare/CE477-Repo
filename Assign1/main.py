import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
#plt.boxplot(data['math score'])
#plt.title("BoxPlot of Math Score")

#plt.boxplot(data['reading score'])
#plt.title("BoxPlot of Reading Score")

#plt.boxplot(data['writing score'])
#plt.title("BoxPlot of Writing Score")

#plt.boxplot(data['writing score'],data['math score'],data ['reading score'])3

plt.rcParams["figure.autolayout"] = True

#showing boxplot of different columns:
#dataFrm = pd.DataFrame({"Math": data['math score'], "Writing": data['writing score'], "Reading": data['reading score']})
#3ax = dataFrm[['Math', 'Writing', 'Reading']].plot(kind='box', title='boxplot')
plt.show()