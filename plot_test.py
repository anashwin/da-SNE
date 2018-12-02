import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

pts = np.loadtxt('dendritic/ts_out/Y_final.txt',skiprows=2)
time_asgn = np.loadtxt('dendritic/assignments.txt',dtype=int)
# time_asgn = np.loadtxt('data_files/components.txt',dtype=int)

# max_T = max(time_asgn)
max_T = max(time_asgn)

net_pts = np.loadtxt('dendritic/full_net_out/Y_final.txt',skiprows=2)
bh_pts = np.loadtxt('dendritic/full_bh_out/Y_final.txt',skiprows=2)

# labels = np.loadtxt('pollen_labels.txt',dtype=int) - 1

# color_key = ['xkcd:azure', 'xkcd:green', 'xkcd:fuchsia', 'xkcd:gold', 'xkcd:lavender', 'xkcd:orange', 'xkcd:black', 'xkcd:maroon', 'xkcd:red', 'xkcd:lime', 'xkcd:grey']
color_key = [(np.random.rand(), np.random.rand(), np.random.rand()) for c in xrange(max_T+1)]
# shape_key = ["o", "D"]

# color_array = np.array([color_key[labels[i]] for i in xrange(len(labels))])
# shape_array = [shape_key[time_asgn[i]] for i in xrange(len(time_asgn))]

fig, (ax_new, ax_net, ax_bh) = plt.subplots(1,3, figsize=(18,6))

for t in xrange(max_T+1): 

    t_pts = pts[time_asgn==t,:]
    # t_color_array = color_array[time_asgn==t]
    
    ax_new.scatter(t_pts[:,0], t_pts[:,1], c=color_key[t])
    ax_new.set_title("Time-series SNE")

    t_net_pts = net_pts[time_asgn==t, :]
    ax_net.scatter(t_net_pts[:,0], t_net_pts[:,1], c=color_key[t])
    ax_net.set_title("Net-SNE")

    t_bh_pts = bh_pts[time_asgn==t, :]
    ax_bh.scatter(t_bh_pts[:,0], t_bh_pts[:,1], c=color_key[t])
    ax_bh.set_title("BH SNE with offset")


fig.savefig("dend_t" + str(max_T) + "_plot_comp_full.png", bbox_inches="tight")

plt.show()
