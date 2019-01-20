import numpy as np
import bh_da_sne
from sklearn.decomposition import PCA

data = np.loadtxt('example_data/pollen.txt',delimiter=',').T

data = np.log(1+data)

data = data - np.mean(data, axis=1, keepdims=True)
data = data/(np.sum(data**2, axis=1, keepdims=True))**.5


pca = PCA(n_components=50)

pc_data = pca.fit_transform(data)

print(pc_data.shape)

embedded,betas = bh_da_sne.run_bh_tsne(pc_data, initial_dims=pc_data.shape[1], theta=0.3,
                                       verbose=True)

np.savetxt('pollen_out/bh_da_out.txt', embedded)
np.savetxt('pollen_out/betas.txt', betas)
