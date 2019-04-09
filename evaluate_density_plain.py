import sys
import numpy as np
from sklearn.linear_model import LinearRegression
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

if len(sys.argv) > 2:
    orig = sys.argv[1]
    flav = sys.argv[2]

    if '.txt' in orig:
        orig = orig[:orig.find('.txt')]
    
    infile = flav + '_' + orig + '_out.txt'
    betafile = flav + '_' + orig + '_betas.txt'
    path = ''
    
elif len(sys.argv) == 2:
    orig = sys.argv[1]
    if '.txt' in orig:
        orig = orig[:orig.find('.txt')]
    
    infile = 'bh_da_' + orig + '_out.txt'
    betafile = 'bh_da_' + orig + '_betas.txt'
    path = '' 
    flav = 'bh_da'
    # infile = sys.argv[1]
    # betafile = 'bh_da_drastic_no-pca_betas.txt'
    # orig = np.loadtxt('gaussian_density_drastic.txt').T

if len(sys.argv) > 3:
    path = sys.argv[3]

orig = np.loadtxt(path + orig + '.txt')
if orig.shape[0] < orig.shape[1]:
    orig = orig.T

    
plotfile = infile[:infile.find('out')] + 'plot_plain.png'

subsample = .1

# orig = np.loadtxt('rect_K2_overlap.txt').T

embedded = np.loadtxt(infile)
N = embedded.shape[0]

subsample_sz = int(subsample*N)
indices = np.random.choice(N, subsample_sz, replace=False)
orig = orig[indices,:]
embedded = embedded[indices,:]


# print np.array([sum(v1) for v1 in orig[:10]])
means = np.mean(orig, axis=0)
# print means
orig = orig - np.vstack((means for _ in xrange(subsample_sz)))

orig /= np.max(abs(orig))

orig_D = np.array([[sum((v1-v2)**2) for v2 in orig] for v1 in orig])
DD = np.array([[sum((v1-v2)**2) for v2 in embedded] for v1 in embedded])

# print np.sum(orig_D, axis=1)/orig_D.shape[0]
# print orig[0], orig[300]
# print orig_D[:10,300:310]

eps = 1.e-12

# Q = (1+DD)**(-1)

# Q /= (np.sum(Q))

# Finding nearest neighbors

NNs = 50

np.fill_diagonal(orig_D, np.inf)
np.fill_diagonal(DD, np.inf)
# orig_D[orig_D == 0.] = np.inf
# DD[orig_D == 0.] = np.inf

betas = np.loadtxt(betafile)
betas = betas[indices]
# print betas

P = np.exp(-betas*orig_D)

P /= np.sum(P,axis=0)

P = 1./(2.*P.shape[0])*(P+P.T)

P_sort = np.sort(orig_D, axis=0)
P_min = np.min(P_sort)

orig_D[orig_D == np.inf] = 0.

P_marg = np.sum(P, axis=0)

print P_sort
avg_dist = np.sum(np.sqrt(orig_D)*P/P_marg, axis=0)

orig_radii = np.log(np.sum(P_sort[NNs:NNs+1,:],axis=0)/P_min)


Q = (1+DD)**(-1)
Q = Q/np.sum(Q)
marginals = np.sum(Q, axis=0)
# print orig_radii

# taus = np.ones(subsample_sz,dtype=float)
# taus_min = -DBL_MAX*np.ones(subsample_sz,dtype=float)
# taus_max = DBL_MAX*np.ones(subsample_sz,dtype=float)
# errors = np.ones(subsample_sz,dtype=float)
# c_ent = np.log(NNs)*np.ones(subsample_sz,dtype=float)

# eps=1.e-6
# ctr=0
# while any(abs(errors) > eps) and ctr < 100:
#     Q = (1+DD*taus)**(-1)
    
#     Q /= np.sum(Q)

#     # print Q[0,:10]
    
#     marginals = np.sum(Q, axis=0)
    
#     entropies = -np.sum(Q/marginals*np.log(Q/marginals+eps), axis=0)

#     # print max(entropies), min(entropies), max(taus), min(taus)
    
#     errors = entropies - c_ent

#     # print errors[:10]
    
#     # (betas, betas_min, betas_max) = map(modify_sig, betas, betas_min, betas_max, errors)
#     ans = np.array(map(modify_sig, taus, taus_min, taus_max, errors))

#     taus = ans[:,0]

#     # print betas
    
#     taus_min = ans[:,1]
#     taus_max = ans[:,2]
#     # print ans.shape
#     # print blah
#     ctr += 1

DD[DD == np.inf] = 0.
    
avg_embedded_dist = np.sum(np.sqrt(DD)*Q/marginals, axis=0)

Q_sort = np.sort(DD, axis=0)

radii = np.sum(Q_sort[NNs:NNs+1,:],axis=0)

# N_bins = 50
# bin_width = (max(radii) - min(radii))/N_bins

# bins = np.arange(min(radii), max(radii)+bin_width, bin_width)

# plt.hist(radii[0:250], bins=bins, alpha=.8)
# plt.hist(radii[250:], bins=bins, color='green', alpha=.8)

# plt.show()
# plt.clf()
lg_dist = np.log(avg_dist)
lg_emb_dist = np.log(avg_embedded_dist)

plt.scatter(orig_radii, np.log(radii), s=4)

plt.show()
plt.clf()

color_dict = {'bh':'green', 'bh_da':'blue', 'notails':'orange', 'bh_da_init':'magenta'}

fig, ax = plt.subplots(1,1)

ax.scatter(lg_dist, lg_emb_dist, c=color_dict[flav], s=4)


# OUTLIERS
OUTLIER = False
if OUTLIER:
    q25, med, q75 = np.percentile(lg_dist, [25, 50, 75])
    outbig = med + 1.5*(q75-q25)
    outsmall = med - 1.5*(q75-q25)

    lg_dist_clean = lg_dist[lg_dist < outbig]
    lg_emb_dist_clean = lg_emb_dist[lg_dist < outbig]

    lg_emb_dist_clean = lg_emb_dist_clean[lg_dist_clean > outsmall] 
    lg_dist_clean = lg_dist_clean[lg_dist_clean > outsmall] 

else:
    lg_dist_clean = lg_dist
    lg_emb_dist_clean = lg_emb_dist

reg = LinearRegression().fit(lg_dist_clean.reshape(-1,1), lg_emb_dist_clean.reshape(-1,1))

rsq= reg.score(lg_dist_clean.reshape(-1,1), lg_emb_dist_clean.reshape(-1,1))
print rsq

print reg.coef_, reg.intercept_

min_x = min(lg_dist)
max_x = max(lg_dist)

bins = 10
x_pts = np.arange(min_x, max_x + (max_x-min_x)/bins, (max_x-min_x)/bins).reshape(-1,1)
y_pts = reg.predict(x_pts)

ax.plot(x_pts, y_pts, 'r-', lw=2.5)
ax.set_title('$R^2 = {:4.3f}$'.format(rsq))

flav_dict = {'bh':'Original', 'bh_da':'Density-aware', 'notails':'No tails',
             'bh_da_init':"Initialized"}

fig.suptitle('Comparison of Densities (' + flav_dict[flav] + ')')

ax.title.set_fontsize(14)
ax.title.set_fontweight('bold')

ax.set_xlabel('log(Original Density)')
ax.set_ylabel('log(Embedded Density)')

# ax.xaxis.label.set_fontsize(14)
# ax.xaxis.label.set_fontweight('bold')
# plt.xlim(0,50)

plt.show()
fig.savefig('plots/'+plotfile, bbox_inches='tight')
