import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np


# loading data
df = pd.read_csv('data/_8b9c6d9ae39e206610c6fd96894615a5_classification.csv')


# calculating TP, FP, FN & TN
TP = df[(df["pred"] == 1) & (df["true"] == 1)]
FP = df[(df["pred"] == 1) & (df["true"] == 0)]
FN = df[(df["pred"] == 0) & (df["true"] == 1)]
TN = df[(df["pred"] == 0) & (df["true"] == 0)]
print("TP: {0} FP: {1} FN: {2} TN: {3}".format(len(TP), len(FP), len(FN), len(TN)))


# calculating the main quality metrics of the classifier
acc = accuracy_score(df["true"], df["pred"])
pr = precision_score(df["true"], df["pred"])
rec = recall_score(df["true"], df["pred"])
f1 = f1_score(df["true"], df["pred"])
print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nF1_score: {3}".format(acc.__round__(2),
                                                                         pr.__round__(2),
                                                                         rec.__round__(2),
                                                                         f1.__round__(2)))


# loading data 2
df2 = pd.read_csv("data/_eee1b9e8188f61bc35d954fbeb94e325_scores.csv")
df2.head()


# Calculating AUC-ROC for each classifier
clf_names = df2.columns[1:]
scores = pd.Series([roc_auc_score(df2["true"], df2[clf]) for clf in clf_names], index=clf_names)
print(scores.sort_values(ascending=False).index[0])


# precision-recall curve
pr_scores = []
for clf in clf_names:
    pr_curve = precision_recall_curve(df2["true"], df2[clf])
    pr_scores.append(pr_curve[0][pr_curve[1] >= 0.7].max())
print(clf_names[np.argmax(pr_scores)])
