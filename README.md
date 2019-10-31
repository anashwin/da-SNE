This software package will hopefully contain a density-aware modification of the Barnes-Hut implementation of t-SNE by Van Der Maaten

Code is very much in beta stage, I am still making it more easily usable and most functionality is undocumented. 

All of the following is adapted from Van Der Maaten's repository: https://github.com/lvdmaaten/bhtsne

# Installation #

On Linux or OS X, compile the source using the following command:

```
g++ cell.cpp da_sptree.cpp density_sptree.cpp da_sne.cpp da_sne_main.cpp -o bh_da_sne -O2
```

The executable will be called `bh_da_sne`.

# Usage #

Demonstration of usage in Python:

```python
import numpy as np
import bh_da_sne

data = np.loadtxt("mnist2500_X.txt", skiprows=1)

embedding_array, orig_densities, emb_densities = bh_da_sne.run_bh_tsne(data, initial_dims=data.shape[1], theta=.3,
verbose=True, perplexity=50, max_iter = 1000, use_pca=True, Y_samples=None)
```
You can also use my pre-wrtten wrapper:

```
python run_da_python.py input.txt
```

which will generate the embedding into the file `bh_dagrad_input_out.txt'; the length-scales
will be in `bh_dagrad_input_betas.txt', the original and embedded local radii will be
saved as `bh_deagrad_input_orig_margD.txt` and `bh_dagrad_input_orig_embD.txt` respectively. 