import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
#batch_size = 3
#batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

df = pd.read_csv('data/4.1.3_wine_quality.csv', sep=';')
pca = PCA(n_components=2)
fit = pca.fit(df)
print(pca.explained_variance_ratio_)
# For PCA variable 1
#   --> Note : np.dot(pca.components_[1],pca.components_[1]) == 1, so this is
#              what we want to use
print("pc1 : {} : {}".format(df.columns, pca.components_[0]**2))
# For PCA variable 2
print("pc2 : {} : {}".format(df.columns, pca.components_[1]**2))

### Seems like 'free_sulfur_dioxide' and 'total_sulfur_dioxide' are the determining
### components
wineq = torch.from_numpy(df.values)
print(wineq.shape, wineq.dtype)

data=wineq[:,:-1]
print(data, data.shape)

target = wineq[:, -1]
print(target, target.shape)



sys.exit(0)
