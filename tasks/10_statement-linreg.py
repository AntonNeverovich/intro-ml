import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# loading data
train = pd.read_csv("data/salary-train.csv")


# data preparation
def text_preparation(text: pd.Series) -> pd.Series:
    return text.str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)


# text to feature vectors transformation
vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_preparation(train["FullDescription"]))


# replacing the blanks in the columns LocationNormalized and ContractTime on the special string 'nan'
train["LocationNormalized"].fillna("nan", inplace=True)
train["ContractTime"].fillna("nan", inplace=True)


# one-hot-coding features
enc = DictVectorizer()
X_train_cat = enc.fit_transform(train[["LocationNormalized", "ContractTime"]].to_dict("records"))


# combining all the signs into one matrix "objects-attributes"
X_train = hstack([X_train_text, X_train_cat])


# ridge regression
y_train = train["SalaryNormalized"]
model = Ridge(alpha=1, random_state=241)
fit = model.fit(X_train, y_train)
print(fit)


# building forecasts
test = pd.read_csv("data/_d0f655638f1d87a0bdeb3bad26099ecd_salary-test-mini.csv")

X_test_text = vec.transform(text_preparation(test["FullDescription"]))
X_test_cat = enc.transform(test[["LocationNormalized", "ContractTime"]].to_dict("records"))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print('{}  {}'.format(y_test[0].__round__(2), y_test[1].__round__(2)))
