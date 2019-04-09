import sys
import numpy as np
import bhtsne

from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
sub = False
subsample = 1.

# data = np.loadtxt('../example_data/pollen.txt',delimiter=',').T

# data = np.log(1+data)

# data = data - np.mean(data, axis=1, keepdims=True)
# data = data/(np.sum(data**2, axis=1, keepdims=True))**.5

# pca = PCA(n_components=50)

# pc_data = pca.fit_transform(data)

infile = sys.argv[1]
path = ''
out_path = ''
title = ''
if len(sys.argv) > 2:
    title = sys.argv[2]
if len(sys.argv) > 3: 
    path = sys.argv[3]
if len(sys.argv) > 4:
    subsample = float(sys.argv[4])
if len(sys.argv) > 5: 
    out_path = sys.argv[5]

# outfile = sys.argv[2]
# betafile = sys.argv[3]

if subsample != 1.:
    sub = True

pcafile = title + '_' + infile + '_pca'
outfile = out_path + 'bh_' + pcafile + '_out.txt'
betafile = out_path + 'bh_' + pcafile + '_betas.txt'

pcafile = out_path + pcafile


pc_data_npz = np.load(path + infile+'.npz')
if 'data' in pc_data_npz.files: 
    pc_data = np.log(1+pc_data_npz['data'])
    pc_indices = pc_data_npz['indices']
    pc_indptr = pc_data_npz['indptr']
    pc_shape = pc_data_npz['shape']

    sparse_data = sparse.csr_matrix((pc_data, pc_indices, pc_indptr), pc_shape)

    print sparse_data.shape
    
    tsvd = TruncatedSVD(n_components = 20)

    transformed = tsvd.fit_transform(sparse_data)
else:
    data = np.log(1+pc_data_npz['X'])

    print pc_data_npz['genes'].shape, data.shape
    tsvd = TruncatedSVD(n_components = 20)

    transformed = tsvd.fit_transform(data)

    
N,D = transformed.shape
if sub:
    sub_sz = int(subsample*N)
    indices = np.random.choice(N, sub_sz, replace=False)
    transformed = transformed[indices,:]

np.savetxt(pcafile + '.txt', transformed)

embedded, betas = bhtsne.run_bh_tsne(transformed, initial_dims=transformed.shape[1], theta=0.3,
                                     verbose=True, perplexity=50, max_iter=1000, use_pca=False)

print embedded.shape, betas.shape
np.savetxt(outfile , embedded)
np.savetxt(betafile, betas)
