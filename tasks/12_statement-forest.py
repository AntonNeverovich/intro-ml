import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from utils import write_output_file as w


# loading data
df = pd.read_csv("data/_8d955d45315ff739d75fd4de3c97acf9_abalone.csv")

# conversion trait 'sex' to numeric
df["Sex"].replace({"F": -1, "I": 0, "M": 1}, inplace=True)

# division the contents of the files into attributes and a target variable
X = df.loc[:, "Sex":"ShellWeight"]
y = df["Rings"]

# training a random forest
cv = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for n in range(1, 51):
    model = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
    scores.append(score)

# determination at what minimum number of trees a random forest shows a cross-validation quality above 0.52
for n, score in enumerate(scores):
    if score > 0.52:
        w.write_answer('Task #25. Random Forest', str(n + 1))
        break
