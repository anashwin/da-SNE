# Generate a mixture of K Gaussians with different variances

import numpy as np

D = 3

K = 2

sep = 1.
means = np.array([[(-1)**(j+k)*k*sep for j in xrange(D)] for k in xrange(K)])

growth = 1.
base = .5

vars = np.array([growth**i*base*np.eye(D) for i in xrange(K)])

# print type(means[0,:])

N = 500

weights = np.array([1./K for k in xrange(K)])

N_pts = N*weights
N_pts = map(int, N_pts)

samples = np.zeros((N, D))

ct = 0
for k in xrange(K):
    
    k_samples = np.random.multivariate_normal(means[k,:], vars[k], N_pts[k])

    samples[ct:ct+N_pts[k]] = k_samples
    ct+=N_pts[k]

np.savetxt('gaussian_density_overlap.txt', samples.T, delimiter='\t')
