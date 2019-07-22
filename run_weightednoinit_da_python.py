import sys
import numpy as np
import bh_da_sne_init
from sklearn.decomposition import PCA

infile = sys.argv[1]
if '.txt' in infile:
    infile = infile[:infile.find('.txt')]

file_root = 'bh_dagradnoinitplus_{}_{}_{}.txt'

# outfile = 'bh_da_' + infile + '_out.txt'
# betafile = 'bh_da_' + infile + '_betas.txt'
# orig_d_file = 'bh

weights = np.array([2.**(i/5.) for i in xrange(-5, 6)])

max_iter = 1000

pc_data = np.loadtxt(infile+'.txt').T

if pc_data.shape[0] < pc_data.shape[1]:
    pc_data = pc_data.T

print(pc_data.shape)

# DROPOUT
## sums = np.sum(pc_data**2, axis=1)
## print(min(sums))

## pc_data = pc_data[sums > .1, :]

## print(pc_data.shape)

# Y_file = 'bh_' + infile + '_out.txt'
# Y_samples = np.loadtxt(Y_file)

# DEBUG PRESENCE OF INITIAL SAMPLES
# weights = [1.0, 2.0, 3.0]
for i, w in enumerate(weights): 
    embedded, betas, orig_densities, emb_densities = bh_da_sne_init.run_bh_tsne(pc_data, 
                                                                                initial_dims=pc_data.shape[1], theta=0.3,thresh=1000.0, verbose=True, perplexity=30, max_iter=max_iter, 
                                                                                use_pca=False, Y_samples = None, weight = w)

    print embedded.shape, betas.shape 
    np.savetxt(file_root.format(infile, 'out', w), embedded)
    np.savetxt(file_root.format(infile, 'betas', w), betas)
    np.savetxt(file_root.format(infile, 'marg_origD', w), orig_densities)
    np.savetxt(file_root.format(infile, 'marg_embD', w), emb_densities)
