import numpy as np
import matplotlib.pyplot as plt



# Goal: Figure out how the DA algorithms deal with overlaps?

D = 2
N = 250
K = 2

series1 = np.random.rand(N,D)

series2 = np.random.rand(N,D)

long_len = 10.
short_len = 1.

dil1 = np.array([long_len, short_len])
trans1 = np.vstack([long_len/2., short_len/2.] for _ in xrange(N))

dil2 = np.array([short_len, long_len])
trans2 = np.vstack([short_len/2., long_len/2.] for _ in xrange(N))

series1 = dil1*series1 - trans1

series2 = dil2*series2 - trans2


plt.scatter(series1[:,0], series1[:,1])
plt.scatter(series2[:,0], series2[:,1])

plt.show()

data = np.vstack((series1, series2)).T

print data.shape

np.savetxt('rect_K2_overlap.txt', data)
