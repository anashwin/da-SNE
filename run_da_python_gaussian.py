import sys
import numpy as np
import bh_da_sne
from sklearn.decomposition import PCA

# data = np.loadtxt('../example_data/pollen.txt',delimiter=',').T

# data = np.log(1+data)

# data = data - np.mean(data, axis=1, keepdims=True)
# data = data/(np.sum(data**2, axis=1, keepdims=True))**.5

# pca = PCA(n_components=50)

# pc_data = pca.fit_transform(data)

infile = sys.argv[1]
outfile = sys.argv[2]
betafile = sys.argv[3]

pc_data = np.loadtxt(infile).T

print(pc_data.shape)

embedded, betas = bh_da_sne.run_bh_tsne(pc_data, initial_dims=pc_data.shape[1], theta=0.3,
                                        thresh=1.0, verbose=True, perplexity=50, max_iter=1000)

print embedded.shape, betas.shape
np.savetxt(outfile , embedded)
np.savetxt(betafile, betas)
