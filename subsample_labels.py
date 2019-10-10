# Takes in a set of indices that are subsampled and generates a label file with the labels for
# those indices

import numpy as np
import sys

index_file = sys.argv[1]
label_file = sys.argv[2]

delim = ','
if len(sys.argv) > 3:
    delim = sys.argv[3]

indices = np.loadtxt(index_file, dtype=float).astype(int)

labels = np.loadtxt(label_file, dtype=str, delimiter=delim)

if len(labels.shape) > 1:
    out_labels = labels[indices, :]
    N = labels.shape[0]
else:
    out_labels = labels[indices]
    N = len(labels)

print len(indices), len(labels)


subsample = 1.*len(indices) / N

out_label_file = label_file[:label_file.find('.txt')] + '_{:.3f}.txt'.format(subsample)
    
np.savetxt(out_label_file, out_labels, fmt='%s')
