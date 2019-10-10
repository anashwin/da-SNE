import numpy as np
import sys

indir = 'data/'

outdir = 'data/'

infile = sys.argv[1]
subsample = float(sys.argv[2])



if len(sys.argv) > 3:
    labelfile = sys.argv[3]

    if '.csv' in labelfile:
        labelfile = labelfile[:labelfile.find('.csv')]

    labels = np.loadtxt(indir + labelfile + '.csv',dtype=str, delimiter=',')


if '.txt' in infile:
    infile = infile[:infile.find('.txt')]

data = np.loadtxt(indir + infile + '.txt')

N = data.shape[0]

if subsample < 1: 
    out_N = int(subsample * N)
else:
    out_N = int(subsample)
    subsample = (1.*out_N) / N

print out_N

indices = np.random.choice(N, out_N, replace=False)
out_data = data[indices, :]

out_root = '{}_{}{:.3f}.txt'

out_indfile = out_root.format(infile, 'ind_', subsample)

np.savetxt(outdir + out_indfile, indices)

out_datafile = out_root.format(infile,'',subsample)

np.savetxt(outdir + out_datafile, out_data)

if len(sys.argv) > 3:
    out_labelfile = out_root.format(labelfile,'', subsample)
    if len(labels.shape) > 1: 
        out_labels = labels[indices,:]
    else:
        out_labels = labels[indices]

    np.savetxt(outdir + out_labelfile, out_labels, fmt='%s')
