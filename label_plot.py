import numpy as np
import sys

import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt

infile =  sys.argv[1]
labelfile = sys.argv[2]


labels = np.loadtxt(labelfile, dtype=str, delimiter=' ')
# labels = np.array(map(int, labels))

if len(labels.shape) > 1:
    if len(sys.argv) > 3:
        labels = labels[:,int(sys.argv[3])]
    else: 
        labels = labels[:,-1]

print labels.shape
        
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

    pts = np.loadtxt(infile)

    while '/' in infile:
        infile = infile[infile.find('/') + 1:]
    
    outfile = infile[:infile.find('.txt')] + '_embedding.png'
else:
#    if 'out' in infile: 
#        outfile = infile[:infile.find('out')] + '_embedding.png'
#    else:

    pts = np.loadtxt(infile+'.txt')
    while '/' in infile:
        infile = infile[infile.find('/') + 1:]

    outfile = infile + '_embedding.png'
    
if pts.shape[0] < pts.shape[1]:
    pts = pts.T
    
if len(sys.argv) > 3: 
    pts = pts[:len(indices),:]

    
fig, ax = plt.subplots(1,1)

label_set = set(labels)

maxL = len(label_set)

print maxL

label_dict = dict()
colors = cm.rainbow(np.linspace(0,1, maxL+1))

for i, label in enumerate(label_set):
    label_dict[label] = i

    sub_pts = pts[labels == label,:]
    ax.scatter(sub_pts[:,0], sub_pts[:,1], color=colors[i], s=2, label=label)


# color_asgn = [colors[label_dict[label]] for label in labels]

# ax.scatter(pts[:,0], pts[:,1], color=color_asgn, s=4)
ax.legend()
plt.show()
# fig.savefig('plots/' + outfile, bbox_inches='tight')
print outfile
