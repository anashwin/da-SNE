This software package will hopefully contain a density-aware modification of the Barnes-Hut implementation of t-SNE by Van Der Maaten

All of the following is adapted from Van Der Maaten's repository: https://github.com/lvdmaaten/bhtsne

# Installation #

On Linux or OS X, compile the source using the following command:

```
g++ da_sptree.cpp da_sne.cpp da_sne_main.cpp -o bh_da_sne -O2
```

The executable will be called `bh_da_sne`.

# Usage #

The code comes with wrappers for Matlab and Python. (Right now, the Matlab wrapper is not implemented for da-SNE!). These wrappers write your data to a file called `data.dat`, run the `bh_tsne` binary, and read the result file `result.dat` that the binary produces. There are also external wrappers available for [Torch](https://github.com/clementfarabet/manifold), [R](https://github.com/jkrijthe/Rtsne), and [Julia](https://github.com/zhmz90/BHTsne.jl). Writing your own wrapper should be straightforward; please refer to one of the existing wrappers for the format of the data and result files.

Demonstration of usage in Matlab:

```matlab
filename = websave('mnist_train.mat', 'https://github.com/awni/cs224n-pa4/blob/master/Simple_tSNE/mnist_train.mat?raw=true');
load(filename);
numDims = 2; pcaDims = 50; perplexity = 50; theta = .5; alg = 'svd';
map = fast_tsne(digits', numDims, pcaDims, perplexity, theta, alg);
gscatter(map(:,1), map(:,2), labels');
```

Demonstration of usage in Python:

```python
import numpy as np
import bh_da_sne

data = np.loadtxt("mnist2500_X.txt", skiprows=1)

embedding_array = bh_da_sne.run_bh_tsne(data, initial_dims=data.shape[1])
```

### Python Wrapper

Usage:

```bash
python bh_da_sne.py [-h] [-d NO_DIMS] [-p PERPLEXITY] [-t THETA] [-e THRESH] 
                  [-r RANDSEED] [-n INITIAL_DIMS] [-v] [-i INPUT]
                  [-o OUTPUT] [--use_pca] [--no_pca] [-m MAX_ITER]
```

Below are the various options the wrapper program `bhtsne.py` expects:

- `-h, --help`                      show this help message and exit
- `-d NO_DIMS, --no_dims`           NO_DIMS
- `-p PERPLEXITY, --perplexity`     PERPLEXITY
- `-t THETA, --theta`               THETA
- `-e THRESH, --thresh`             threshold for betas 
- `-r RANDSEED, --randseed`         RANDSEED
- `-n INITIAL_DIMS, --initial_dims` INITIAL_DIMS
- `-v, --verbose`
- `-i INPUT, --input`               INPUT: the input file, expects a TSV with the first row as the header.
- `-o OUTPUT, --output`             OUTPUT: A TSV file having each row as the `d` dimensional embedding.
- `--use_pca`
- `--no_pca`
- `-m MAX_ITER, --max_iter`         MAX_ITER

