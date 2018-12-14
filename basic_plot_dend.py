import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def int2color(i):
    if i == 0:
        return 'blue'
    else:
        return 'green'
pts = np.loadtxt('da_out_gaussian_drastic.txt')
# pts = np.loadtxt('UMAP_test.txt')
# pts = np.loadtxt('gaussian_density_drastic.txt').T
# asgn = np.loadtxt('example_data/pollen_labels.txt', dtype=int)

color_int = np.array([i/250 for i in xrange(len(pts))],dtype=int)
# max_T = max(asgn)

# color_key = [(np.random.rand(), np.random.rand(), np.random.rand()) for c in xrange(max_T+1)]


# colors = []
# for a in asgn:
#     colors.append(color_key[a])
# colors =np.array(colors)

colors = map(int2color, color_int)

fig, ax = plt.subplots(1,1)

ax.scatter(pts[:,0], pts[:,1], c=colors)

# ax.set_xlim(-60, 60)
# ax.set_ylim(-100, 100)

plt.show()

fig.savefig('da__bh_gaussian_plot.png', bbox_inches='tight')
