import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
# Bar chart with skinThickness against BMI
plt.scatter(data['math score'], data['reading score'])

# Adding Title to the Plot
plt.title("Scatter Plot")

# Setting the X and Y labels
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.show()
