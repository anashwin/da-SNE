# Generate a mixture of K Gaussians with different variances

import numpy as np

flavor = 'drastic'
D = 10

K = 10

sep = 100.
means = np.array([[(-1)**(j+k)*k*sep for j in xrange(D)] for k in xrange(K)])
# means = np.array([[k*sep for j in xrange(D)] for k in xrange(K)])
# means = np.array([[k for k in xrange(K)], [0 for k in xrange(K)], [0 for k in xrange(K)]]).T
# means = np.zeros((K, D))

growth = 2.
base = .1

vars = np.array([growth**i*base*np.eye(D) for i in xrange(K)])

# print type(means[0,:])

N = 3000

weights = np.array([1./K for k in xrange(K)])

N_pts = N*weights
N_pts = map(int, N_pts)

samples = np.zeros((N, D))

ct = 0
for k in xrange(K):
    
    k_samples = np.random.multivariate_normal(means[k,:], vars[k], N_pts[k])

    samples[ct:ct+N_pts[k]] = k_samples
    ct+=N_pts[k]

np.savetxt('gd_D{:d}_K{:d}_G{:2.1f}_{:s}.txt'.format(D,K,growth,flavor),
           samples.T, delimiter='\t')
