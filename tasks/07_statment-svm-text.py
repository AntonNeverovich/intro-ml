import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC


# loading data
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target


# calculating TF-IDF features for all texts
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# choosing the minimum best C-parameter
grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel="linear", random_state=241)
gs = GridSearchCV(model, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
gs.fit(X, y)

C = gs.best_params_.get('C')


# training the SVM across the sample with the optimal parameter C
model = SVC(C=C, kernel="linear", random_state=241)
model.fit(X, y)


# finding 10 words with the highest absolute weight value
words = np.array(vectorizer.get_feature_names())
word_weights = pd.Series(model.coef_.data, index=words[model.coef_.indices], name="weight")
word_weights.index.name = "word"

top_words = word_weights.abs().sort_values(ascending=False).head(10)
print(top_words)


file = open('answers/Task #15. Statement SVM - text.txt', 'w')
for n in top_words['word']:
    file.write(n + ' ')
file.close()
