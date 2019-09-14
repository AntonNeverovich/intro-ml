import pandas as pd
from sklearn.tree import tree

data = pd.read_csv("data/_ea07570741a3ec966e284208f588e50e_titanic.csv", index_col='PassengerId')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data = data.dropna()
data.loc[data['Sex'] != 'female', 'Sex'] = 0
data.loc[data['Sex'] == 'female', 'Sex'] = 1

X = data[['Pclass', 'Fare', 'Age', 'Sex']]
Y = data['Survived']

clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)
importances = clf.feature_importances_


print importances.round(4)