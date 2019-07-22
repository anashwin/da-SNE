import sys
import numpy as np
import da_notails_sne
from sklearn.decomposition import PCA

# data = np.loadtxt('../example_data/pollen.txt',delimiter=',').T

# data = np.log(1+data)

# data = data - np.mean(data, axis=1, keepdims=True)
# data = data/(np.sum(data**2, axis=1, keepdims=True))**.5

# pca = PCA(n_components=50)

# pc_data = pca.fit_transform(data)

infile = sys.argv[1]

if '.txt' in infile:
    infile = infile[:infile.find('.txt')]

outfile = 'notails_' + infile + '_out.txt'
betafile = 'notails_' + infile + '_betas.txt'

# outfile = sys.argv[2]

pc_data = np.loadtxt(infile+'.txt').T

if pc_data.shape[0] < pc_data.shape[1]:
    pc_data = pc_data.T

print(pc_data.shape)

embedded, betas = da_notails_sne.run_bh_tsne(pc_data, initial_dims=pc_data.shape[1], theta=0.3,
                                             verbose=True, perplexity=30, max_iter=1000, use_pca=False)

np.savetxt(outfile , embedded)
np.savetxt(betafile, betas)