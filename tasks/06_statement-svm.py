import pandas as pd
from sklearn.svm import SVC
import numpy as np

# loading data
df = pd.read_csv('data/_f6284c13db83a3074c2b987f714f24f5_svm-data.csv', header=None)
y = df[0]  # target variable
X = df[[1, 2]]  # features
print(df)


# training the classifier with a linear kernel
model = SVC(kernel="linear", C=100000, random_state=241)
model.fit(X, y)


# rooms basic facilities
file = open('answers/Task #14. Statement SVM.txt', 'w')
for n in np.sort(model.support_):
    file.write(str(n+1) + ' ')
file.close()
