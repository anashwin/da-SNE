import numpy as np
import csv

a = np.load('/scratch1/brianhie/scanorama_data/293t_jurkat/293t/tab.npz')

b = np.loadtxt('/scratch1/brianhie/scanorama_data/293t_jurkat/293t/tab.genes.txt', dtype=str)
print a['shape']

print b.shape

print b[:10]
