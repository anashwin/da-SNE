import sys
import numpy as np

fname = 'pollen_P.dat'

with open(fname, 'rb') as f:
    for line in f:
        print(float(line.rstrip()))
