import sys
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

flav = 'bh_dagradnoinit'
orig = sys.argv[1]

if len(sys.argv) > 2:
    orig = sys.argv[1]
    flav = sys.argv[2]

if '.txt' in orig:
    orig = orig[:orig.find('.txt')]

# orig_D_fname = flav + '_' + orig + '_origD.txt'
# emb_D_fname = flav + '_' + orig + '_embD.txt'

emb_fname = flav + '_' + orig + '_out_p{}.txt'

orig_D_fname = flav + '_' + orig + '_marg_origD_p{}.txt'
# orig_D_fname = flav + '_' + orig + '_embD.txt'
emb_D_fname = flav + '_' + orig + '_marg_embD_p{}.txt'
# emb_D_fname = flav + '_' + orig + '_origD.txt'

plotfile = flav + '_' + orig + '_full_plt_p{}.png'

for i in xrange(10):
    perp = 10*(i+1)
    print "Perp = {}".format(perp)

    embedding = np.loadtxt(emb_fname.format(perp))

    lg_dist = np.log(np.loadtxt(orig_D_fname.format(perp)))
    # lg_dist = np.loadtxt(orig_D_fname)
    lg_emb_dist = np.log(np.loadtxt(emb_D_fname.format(perp)))
    # lg_emb_dist = np.loadtxt(emb_D_fname)

    color_dict = {'bh':'green', 'bh_da':'blue', 'notails':'orange', 'bh_da_init':'magenta',
                  'bh_dagradnoinit':'blue'}
    
    # num_bins = 100
    # n, bins, patches = plt.hist(lg_dist, num_bins)

    # plt.show()
    # plt.cla()

    fig, ax = plt.subplots(1,2)

    if flav not in color_dict: 
        flav = 'bh_da'
 
    ax[0].scatter(embedding[:,0], embedding[:,1], c=color_dict[flav], s=1)
    ax[1].scatter(lg_dist, lg_emb_dist, c=color_dict[flav], s=4)

    # OUTLIERS
    OUTLIER = False
    if OUTLIER:
        q25, med, q75 = np.percentile(lg_dist, [0, 50, 99])
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

    ax[1].plot(x_pts, y_pts, 'r-', lw=2.5)
    ax[1].set_title('$R^2 = {0:4.3f}$, Perp = ${1}$'.format(rsq, perp))

    flav_dict = {'bh':'Original', 'bh_da':'Density-aware', 'notails':'No tails',
                 'bh_da_init':"Initialized", 'bh_dagradnoinit':'Gradient density'}

    if flav not in flav_dict: 
        flav = 'bh_da'

    fig.suptitle('Comparison of Densities (' + flav_dict[flav] + ')')

    ax[1].title.set_fontsize(14)
    ax[1].title.set_fontweight('bold')

    ax[1].set_xlabel('log(Original Density)')
    ax[1].set_ylabel('log(Embedded Density)')

# ax.xaxis.label.set_fontsize(14)
# ax.xaxis.label.set_fontweight('bold')
# plt.xlim(0,50)

    plt.show()
    fig.savefig('plots/'+plotfile.format(perp), bbox_inches='tight')
