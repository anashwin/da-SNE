import numpy as np
from matplotlib import pyplot as plt

DBL_MAX = 1.e12
DBL_MIN = 1.e-12
eps = 1e-9


def modify_sig(beta, beta_min, beta_max, delt):

    if abs(delt) < eps:
        return [beta, beta_min, beta_max]
    
    if delt > 0:
        if beta_max == DBL_MAX or beta_max == -DBL_MAX:
            return [2*beta, beta, beta_max]
        else: 
            return [(beta + beta_max)/2, beta, beta_max]
    else:
        if beta_min == DBL_MAX or beta_min == -DBL_MAX:
            return [beta/2., beta_min, beta]
        else: 
            return [(beta + beta_min)/2, beta_max, beta]

# def sq_euclidean_distance(v1, v2):
#     return (v1-v2)**2

embedded = np.loadtxt('bh_da_rect_K2_overlap_out.txt')
# embedded = np.loadtxt('../bhtsne-master/bh_rect_K2_overlap_out.txt')
# embedded = np.loadtxt('../bhtsne-master/pollen_out/bh_out.txt')

DD = np.array([[sum((v1-v2)**2) for v2 in embedded] for v1 in embedded])
N = embedded.shape[0]

c_ent = .01*np.ones(N, dtype=float)

for n in xrange(N):
    DD[n,n] = np.inf

betas = np.ones(N, dtype=float)
betas_min = -DBL_MAX*np.ones(N, dtype=float)
betas_max = DBL_MAX*np.ones(N, dtype=float)
errors = np.ones(N,dtype=float)

eps = 1.e-12
ctr = 0

# while any(abs(errors) > eps) and ctr < 100:
#     Q = (1+DD*betas)**(-1)
#     Q /= np.sum(Q)

#     # print Q[0,:10]
    
#     marginals = np.sum(Q, axis=0)
    
#     entropies = -np.sum(Q*np.log(Q/marginals+eps), axis=0)

#     print max(entropies), min(entropies), max(betas), min(betas)
    
#     errors = entropies - c_ent

#     # print errors[:10]
    
#     # (betas, betas_min, betas_max) = map(modify_sig, betas, betas_min, betas_max, errors)
#     ans = np.array(map(modify_sig, betas, betas_min, betas_max, errors))

#     betas = ans[:,0]

#     # print betas
    
#     betas_min = ans[:,1]
#     betas_max = ans[:,2]
#     # print ans.shape
#     # print blah
#     ctr += 1

Q = (1+DD)**(-1)

Q /= (np.sum(Q)/2)

marginals = np.sum(Q, axis=0)

entropies = -np.sum(Q*np.log(Q/marginals+eps), axis=0)

N_bins = 20.
bin_width = (max(entropies) - min(entropies))/N_bins
bins = np.arange(min(entropies), max(entropies) + bin_width, bin_width)

plt.hist(entropies[0:250], bins=bins, alpha=.5)
plt.hist(entropies[250:], bins=bins, color='green', alpha=.5)
# plt.hist(entropies, bins=bins)

plt.show()

# Q_vals = [[],[],[]]

# ctr = 0
# for i in xrange(N):
#     for j in xrange(i,N):
#         if j < (N/2):
#             if i < (N/2):
#                 Q_vals[0].append(Q[i,j])
#         else:
#             if i < (N/2):
#                 Q_vals[2].append(Q[i,j])
#             else:
#                 Q_vals[1].append(Q[i,j])
# plt.clf()

# print len(Q_vals[0]), len(Q_vals[1]), len(Q_vals[2])

# # print max(Q_vals), min(Q_vals)

# # bin_width = (max(Q_vals) - min(Q_vals))/N_bins
# # bins = np.arange(min(Q_vals), max(Q_vals)+bin_width, bin_width)
# plt.hist(Q_vals, bins=100, stacked=True, color=['red','green','blue'])

# plt.ylim(0,5000)

plt.clf()

lg_betas = np.log(np.loadtxt('bh_da_rect_K2_overlap_betas.txt'))

plt.hist([lg_betas[:250], lg_betas[250:]], bins=50, stacked=True)
# plt.hist(lg_betas, bins=20)

plt.show()

plt.clf()

plt.scatter(lg_betas, entropies)

plt.show()
