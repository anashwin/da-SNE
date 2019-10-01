import numpy as np
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def int2color(i):
    if i == 0:
        return 'blue'
    elif i == 1: 
        return 'green'
    elif i == 2:
        return 'orange'
    elif i == 3:
        return 'black'
    elif i == 4:
        return 'red'
    elif i == 5:
        return 'yellow'
    elif i == 6:
        return 'gray'
    elif i == 7:
        return 'pink'
    elif i == 8:
        return 'turquoise'
    elif i == 9:
        return 'brown'

if len(sys.argv) > 1: 
    infile = sys.argv[1]
    # outfile = sys.argv[2]
    outfile = infile[:infile.find('out')] + 'embedding.png'
else:
    infile = 'bh_da_drastic_out.txt'
    outfile = 'bh_da_drastic_plot.png'

if '.txt' in infile:
    pts = np.loadtxt(infile)
else: 
    pts = np.loadtxt(infile+'.txt')


if pts.shape[0] < pts.shape[1]:
    pts = pts.T
print pts.shape

# pts = np.loadtxt('UMAP_test.txt')
# pts = np.loadtxt('gaussian_density_drastic.txt').T
# asgn = np.loadtxt('example_data/pollen_labels.txt', dtype=int)

if 'bh_da' in infile:
    flav = 'bh_da'
elif 'notail' in infile:
    flav = 'notails' 
else:
    flav = 'bh'
    
N = len(pts)
K = 2
if len(sys.argv) > 2:
    K = int(sys.argv[2])


color_int = np.array([i/(N/K) for i in xrange(len(pts))],dtype=int)
# max_T = max(asgn)

# color_key = [(np.random.rand(), np.random.rand(), np.random.rand()) for c in xrange(max_T+1)]


# colors = []
# for a in asgn:
#     colors.append(color_key[a])
# colors =np.array(colors)

colors = map(int2color, color_int)

flav_dict = {'bh_da':'Density-aware', 'bh':'Original', 'notails':'No tails', 'bh_grad': 'Gradient'}

fig, ax = plt.subplots(1,1)

ax.set_title("Embedding: ({})".format(flav_dict[flav]))

ax.scatter(pts[:,0], pts[:,1], c=colors, s=10)

# ax.set_xlim(-60, 60)
# ax.set_ylim(-100, 100)

plt.show()

print outfile

# fig.savefig('plots/' + outfile, bbox_inches='tight')

