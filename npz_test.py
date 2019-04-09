import numpy as np
import csv

a = np.load('/scratch1/brianhie/scanorama_data/293t_jurkat/jurkat_293t_50_50/tab.npz')

print a.files

b = np.loadtxt('/scratch1/brianhie/scanorama_data/293t_jurkat/293t/tab.genes.txt', dtype=str)
print a['genes']

print b.shape

print b[:10]
