import numpy as np
import sys

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt

infile =  sys.argv[1]
labelfile = sys.argv[2]

labels = np.loadtxt(labelfile, dtype=str)
# labels = np.array(map(int, labels))

subsample = False
flav = 'bh_dagrad'
if len(sys.argv) > 3:
    subsample = sys.argv[3]
    indices = np.array(map(int,np.loadtxt(subsample)))

    indices = indices[indices < len(labels)]
    labels = labels[indices]
    
if '.txt' in infile:
#    if 'out' in infile: 
#        outfile = infile[:infile.find('out')] + '_embedding.png'
#    else:
    outfile = infile[:infile.find('.txt')] + '_embedding.png'

    pts = np.loadtxt(infile)
else:
#    if 'out' in infile: 
#        outfile = infile[:infile.find('out')] + '_embedding.png'
#    else:
    outfile = infile + '_embedding.png'

    pts = np.loadtxt(infile+'.txt')

if pts.shape[0] < pts.shape[1]:
    pts = pts.T
    
if len(sys.argv) > 3: 
    pts = pts[:len(indices),:]

    
fig, ax = plt.subplots(1,1)

label_set = set(labels)

maxL = len(label_set)

label_dict = dict()
for i, label in enumerate(label_set):
    label_dict[label] = i

colors = cm.rainbow(np.linspace(0,1, maxL+1))

color_asgn = [colors[label_dict[label]] for label in labels]

ax.scatter(pts[:,0], pts[:,1], color=color_asgn, s=4)

plt.show()
fig.savefig('plots/' + outfile, bbox_inches='tight')
