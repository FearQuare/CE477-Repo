import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
matrix = data.corr(numeric_only=True)
sn.heatmap(matrix, annot=True)
plt.show()

