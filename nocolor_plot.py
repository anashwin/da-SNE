import numpy as np
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


infile = sys.argv[1]

if '.txt' in infile:
    outfile = infile[:infile.find('out')] + '_embedding.png'
    pts = np.loadtxt(infile)
else:
    outfile = infile[:infile.find('out')] + '_embedding.png'
    pts = np.loadtxt(infile+'.txt')

# pts = np.loadtxt('UMAP_test.txt')
# pts = np.loadtxt('gaussian_density_drastic.txt').T
# asgn = np.loadtxt('example_data/pollen_labels.txt', dtype=int)
# max_T = max(asgn)

# color_key = [(np.random.rand(), np.random.rand(), np.random.rand()) for c in xrange(max_T+1)]


# colors = []
# for a in asgn:
#     colors.append(color_key[a])
# colors =np.array(colors)


fig, ax = plt.subplots(1,1)

ax.scatter(pts[:,0], pts[:,1], s=4)

# ax.set_xlim(-60, 60)
# ax.set_ylim(-100, 100)

plt.show()

fig.savefig('plots/' + outfile, bbox_inches='tight')
