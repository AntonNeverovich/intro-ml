import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from typing import List, Tuple
from utils import write_output_file as w

# loading data
df = pd.read_csv('data/_75fb7a1b6f3431b6217cdbcba2fd30b9_gbm-data.csv')
print(df.head())

X = df.loc[:, "D1":"D1776"].values
y = df["Activity"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


# training GradientBoostingClassifier
def sigmoid(y_pred: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-y_pred))


def log_loss_results(model, X: np.array, y: np.array) -> List[float]:
    return [log_loss(y, sigmoid(y_pred)) for y_pred in model.staged_decision_function(X)]


def plot_loss(learning_rate: float, test_loss: List[float], train_loss: List[float]) -> None:
    plt.figure()
    plt.plot(test_loss, "r", linewidth=2)
    plt.plot(train_loss, "g", linewidth=2)
    plt.legend(["test", "train"])
    plt.show()


min_loss_results = {}
for lr in [1, 0.5, 0.3, 0.2, 0.1]:
    print(f"Learning rate: {lr}")

    model = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
    model.fit(X_train, y_train)

    train_loss = log_loss_results(model, X_train, y_train)
    test_loss = log_loss_results(model, X_test, y_test)
    # plot_loss(lr, test_loss, train_loss)

    min_loss_value = min(test_loss)
    min_loss_index = test_loss.index(min_loss_value) + 1
    min_loss_results[lr] = min_loss_value, min_loss_index

    print(f"Min loss {min_loss_value:.2f} at n_estimators={min_loss_index}\n")

w.write_answer('Task #26. Schedule quality on the test sample', 'overfitting')

min_loss_value, min_loss_index = min_loss_results[0.2]
w.write_answer('Task #27. Minimum log-loss value on the test sample and iteration number',
               f"{min_loss_value:.2f} {min_loss_index}")

# training RandomForestClassifier
model = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]
test_loss = log_loss(y_test, y_pred)

w.write_answer('Task #28. Logloss value on the test is obtained from the random forest', f"{test_loss:.2f}")
