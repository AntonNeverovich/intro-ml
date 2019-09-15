import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# loading data
df_train = pd.read_csv('data/_3abd237d917280ba0d83bfe6bd49776f_perceptron-train.csv', header=None)
X_train = df_train.loc[:, 1:]
y_train = df_train[0]
df_test = pd.read_csv('data/_3abd237d917280ba0d83bfe6bd49776f_perceptron-test.csv', header=None)
X_test = df_test.loc[:, 1:]
y_test = df_test[0]


# training perceptron
model = Perceptron(max_iter=5, tol=None, random_state=241)
model.fit(X_train, y_train)


# calculating accuracy
acc_before = accuracy_score(y_test, model.predict(X_test))
print "acc before = {0}".format(acc_before)


# normalizing training and test samples
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# training preceptron on new samples
model.fit(X_train_scaled, y_train)
acc_after = accuracy_score(y_test, model.predict(X_test_scaled))
print "acc after = {0}".format(acc_after)


# accuracy difference between training and testing samples
diff = acc_after - acc_before
print "difference = {0}".format(diff)