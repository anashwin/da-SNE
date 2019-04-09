import sys
import numpy as np
import bh_da_sne_init
from sklearn.decomposition import PCA

# data = np.loadtxt('../example_data/pollen.txt',delimiter=',').T

# data = np.log(1+data)

# data = data - np.mean(data, axis=1, keepdims=True)
# data = data/(np.sum(data**2, axis=1, keepdims=True))**.5

# pca = PCA(n_components=50)

# pc_data = pca.fit_transform(data)

infile = sys.argv[1]
# outfile = sys.argv[2]
# betafile = sys.argv[3]
if '.txt' in infile:
    infile = infile[:infile.find('.txt')]

outfile = 'bh_da_' + infile + '_out.txt'
betafile = 'bh_da_' + infile + '_betas.txt'

Y_samples = None
max_iter = 1000
if len(sys.argv) > 2:
    Y_samples = np.loadtxt(sys.argv[2])
    max_iter = 500
    
    outfile = 'bh_da_init_' + infile + '_out.txt'
    betafile = 'bh_da_init_' + infile + '_betas.txt'

    
pc_data = np.loadtxt(infile+'.txt').T

if pc_data.shape[0] < pc_data.shape[1]:
    pc_data = pc_data.T

print(pc_data.shape)

embedded, betas = bh_da_sne_init.run_bh_tsne(pc_data, initial_dims=pc_data.shape[1], theta=0.3,
                                             thresh=1000.0, verbose=True, perplexity=30, max_iter=max_iter, use_pca=False, Y_samples = Y_samples)

print embedded.shape, betas.shape
np.savetxt(outfile , embedded)
np.savetxt(betafile, betas)
