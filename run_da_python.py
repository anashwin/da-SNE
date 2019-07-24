import sys
import numpy as np
from bh_da_sne_init import run_bh_tsne
From sklearn.decomposition import PCA

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

file_root = 'bh_dagrad_{}_{}.txt'

# outfile = 'bh_da_' + infile + '_out.txt'
# betafile = 'bh_da_' + infile + '_betas.txt'
# orig_d_file = 'bh


Y_samples = None
max_iter = 1000
if len(sys.argv) > 2:
    Y_samples = np.loadtxt(sys.argv[2])
    max_iter = 500
    file_root = 'init_' + file_root
    max_iter = 250
    # outfile = 'bh_da_init_' + infile + '_out.txt'
    # betafile = 'bh_da_init_' + infile + '_betas.txt'

    
pc_data = np.loadtxt(infile+'.txt').T

if pc_data.shape[0] < pc_data.shape[1]:
    pc_data = pc_data.T

print(pc_data.shape)

# DROPOUT
## sums = np.sum(pc_data**2, axis=1)
## print(min(sums))

## pc_data = pc_data[sums > .1, :]

## print(pc_data.shape)

embedded,betas,orig_densities,emb_densities=run_bh_tsne(pc_data, initial_dims=pc_data.shape[1],
                                                        theta=0.3, verbose=True, perplexity=50,
                                                        max_iter=max_iter, use_pca=False,
                                                        Y_samples = Y_samples, weight=.1)

print embedded.shape, betas.shape, 
np.savetxt(file_root.format(infile, 'out'), embedded)
np.savetxt(file_root.format(infile, 'betas'), betas)
np.savetxt(file_root.format(infile, 'marg_origD'), orig_densities)
np.savetxt(file_root.format(infile, 'marg_embD'), emb_densities)
