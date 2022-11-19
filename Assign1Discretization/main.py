import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
data = pd.read_csv("/Users/yarenkarabacak/Desktop/exams.csv")
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform') # for math score
est_2 = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile') # for reading score
#est.fit(data[['math score']])
#Xt = est.transform(data[['math score']])
new = est.fit_transform(data[['math score']])
new_2=est_2.fit_transform(data[['reading score']])
plt.hist(new_2)
plt.title("Quantile Histogram")
plt.xlabel("Classes")
plt.ylabel("Frequency")

plt.hist(new)
plt.title("Uniform Histogram")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.show()



c_2=csr_matrix(new_2)
a_2 = c_2.toarray()
d_2=c_2.todok()
c=csr_matrix(new)
a = c.toarray()
d=c.todok()
myDictionary=dict(d_2.items())
keys=myDictionary.keys()
values=myDictionary.values()
xaxis=list(c.data)
yaxis=list(data['math score'])
xaxis2=list(c_2.data)








