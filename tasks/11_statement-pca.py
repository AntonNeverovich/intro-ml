import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from utils import write_output_file as w

# loading data
df = pd.read_csv('data/_82b4a9f66c689b3d40dd25ebd761b07f_close_prices.csv')

# training the PCA transformation
X = df.loc[:, "AXP":]
pca = PCA(n_components=10)
pca.fit(X)

sum_var = 0
i_ = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    sum_var += v
    i_ += 1
    if sum_var >= 0.9:
        break
print('Task #21. Quantity of 90 percent dispersion')
print(str(i_))
w.write_answer('Task #22. Quantity of 90 percent dispersion.txt', str(i_))


# application the constructed transformation to the source data
X0 = pd.DataFrame(pca.transform(X))[0]

# loading the Dow Jones information
df2 = pd.read_csv('data/_82b4a9f66c689b3d40dd25ebd761b07f_djia_index.csv')

# calculating Pearson correlation
corr = np.corrcoef(X0, df2["^DJI"])
print('\nTask #23. Pearson correlation')
print(corr[1, 0].__round__(2))
w.write_answer('Task #23. Pearson correlation', str(corr[1, 0].__round__(2)))


# calculating most weight company
company = X.columns[np.argmax(pca.components_[0])]
print('\nTask #24. Which company')
print(company)
w.write_answer('Task #24. Which company', str(company))
