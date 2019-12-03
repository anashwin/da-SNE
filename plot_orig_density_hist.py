import sys
import numpy as np

import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.cm as cm
from matplotlib import pyplot as plt

flav = 'bh_dagrad-dimnu'

indir = 'out/'

infile = sys.argv[1]

labelfile = sys.argv[2]

if len(sys.argv) > 3:
    flav = sys.argv[3]

indata = np.log(np.loadtxt(indir + flav + '_' + infile) + 1e-5)

labels = np.loadtxt(labelfile, dtype=str, delimiter=' ')

if len(labels.shape) > 1:
    labels = labels[:,-1]

nozeros = True
if nozeros:
    # inds = indata > .5
    inds = indata > -10
    indata = indata[inds]
    labels = labels[inds]
    
label_set = set(labels)

maxL = len(label_set)

colors = cm.rainbow(np.linspace(0,1, maxL+1))

out_list = []
label_dict = dict()
color_list = []

bins = 100

plt.figure()

for i, label in enumerate(label_set):
    label_dict[label] = i

    sub_pts = indata[labels == label]
    
    out_list.append(sub_pts)

    color_list.append(colors[i])

    print label, len(sub_pts)
    
print len(out_list), len(color_list)

plt.hist(out_list, bins, color=color_list, stacked=True, normed=True)
# plt.hist(out_list, bins, color=color_list)
plt.show()


# plt.savefig('plots/' + flav + '_' + infile[:infile.find('.txt')]
#             + '_dense_hist.png', bbox_inches='tight')
