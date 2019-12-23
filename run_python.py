import sys
import numpy as np
import bhtsne
from sklearn.decomposition import PCA, TruncatedSVD

# data = np.loadtxt('../example_data/pollen.txt',delimiter=',').T

# data = np.log(1+data)

# data = data - np.mean(data, axis=1, keepdims=True)
# data = data/(np.sum(data**2, axis=1, keepdims=True))**.5

# pca = PCA(n_components=50)

# pc_data = pca.fit_transform(data)

# infile = sys.argv[1]
# outfile = 'bh_' + infile + '_out.txt'
# betafile = 'bh_' + infile + '_betas.txt'

infile = sys.argv[1]
# outfile = sys.argv[2]
# betafile = sys.argv[3]
if '.txt' in infile:
    infile = infile[:infile.find('.txt')]

indir = 'data/'
while '/' in infile:
    indir += infile[:infile.find('/') + 1]
    infile = infile[infile.find('/') + 1:]


outdir = 'out/'

file_root = '{}bh-exp-origD_{}_{}.txt'

# outfile = sys.argv[2]

pc_data = np.loadtxt(indir + infile+'.txt').T

if pc_data.shape[0] < pc_data.shape[1]:
    pc_data = pc_data.T

# truncate = True
truncate = False

if truncate:

    col_sums = pc_data.sum(axis=1)

    good_inds = col_sums > .5* col_sums.mean()
    
    pc_data = pc_data[good_inds, :]

    pc_data = np.log(1 + pc_data)
    
    new_D = pc_data.shape[1] / 2 
    svd = TruncatedSVD(n_components = new_D)

    pc_data = svd.fit_transform(pc_data)

    good_inds = np.arange(len(good_inds))[good_inds]
    np.savetxt(indir + infile + '_filterinds.txt', good_inds)

    file_root = file_root[0:2] + 'trunc_' + file_root[2:]
    print(pc_data.shape)
    
if len(sys.argv) > 2:
    subsample = float(sys.argv[2])

    N_old = pc_data.shape[0]
    N_new = int(N_old * subsample)

    indices = np.random.choice(np.arange(N_old), size=N_new, replace=False)

    pc_data = pc_data[indices, :]
    
print(pc_data.shape)

embedded, betas, orig_densities, emb_densities = bhtsne.run_bh_tsne(pc_data, initial_dims=pc_data.shape[1], theta=0.5,
                              verbose=True, perplexity=70, max_iter=1000, use_pca=False)

# np.savetxt(outfile , embedded)
# np.savetxt(betafile, betas)
np.savetxt(file_root.format(outdir, infile, 'out'), embedded)
np.savetxt(file_root.format(outdir, infile, 'betas'), betas)
np.savetxt(file_root.format(outdir, infile, 'marg_origD'), orig_densities)
np.savetxt(file_root.format(outdir, infile, 'marg_embD'), emb_densities)
