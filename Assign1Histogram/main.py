import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")

plt.hist(data['gender'])
plt.title("Gender Histogram")
plt.xlabel("Gender")
plt.ylabel("Frequency")
#
plt.hist(data['race/ethnicity'])
plt.title("Race/Ethnicity Histogram")
plt.xlabel("Race/Ethnicity")
plt.ylabel("Frequency")
#
plt.hist(data['lunch'])
plt.title("Lunch Histogram")
plt.xlabel("Lunch")
plt.ylabel("Frequency")
#
plt.hist(data['test preparation course'])
plt.title("Test Preparation Course Histogram")
plt.xlabel("Test Preparation Course")
plt.ylabel("Frequency")

plt.rcParams["figure.autolayout"] = True
plt.show()

#another way of showing data with histogram:
#fig = px.histogram(data, x="race/ethnicity")
#fig.show()


