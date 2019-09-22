import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


# load data wine
columns = [
            "Class",
            "Alcohol",
            "Malic acid",
            "Ash",
            "Alcalinity of ash",
            "Magnesium",
            "Total phenols",
            "Flavanoids",
            "Nonflavanoid phenols",
            "Proanthocyanins",
            "Color intensity",
            "Hue",
            "OD280/OD315 of diluted wines",
            "Proline",
        ]
df = pd.read_csv('data/wine.data', index_col=False, names=columns)
# print df.head()


# extract features and classes from the data
X = df.loc[:, df.columns != 'Class']
y = df['Class']
# print y.head()
# print X.head()


# quality assessment 5-block cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# finding the classification accuracy on cross-validation for the k nearest neighbor method
def get_best_score(X, y, cv):
    best_score, best_k = None, None

    for k in range(1, 51):  # k = [1, 50]
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

        if best_score is None or score > best_score:
            best_score, best_k = score, k

    return best_score, best_k


score, k = get_best_score(X, y, cv)
print('k - optimal = {0}\nclassification accuracy = {1}'.format(k, score.round(2)))


# feature scaling
score1, k1 = get_best_score(scale(X), y, cv)
print('k1 - optimal = {0}\nnew classification accuracy = {1}'.format(k1, score1.round(2)))