This software package will hopefully contain a density-aware modification of the Barnes-Hut implementation of t-SNE by Van Der Maaten

All of the following is adapted from Van Der Maaten's repository: https://github.com/lvdmaaten/bhtsne

# Installation #

On Linux or OS X, compile the source using the following command:

```
g++ cell.cpp da_sptree.cpp density_sptree.cpp da_sne.cpp da_sne_main.cpp -o bh_da_sne -O2
```

The executable will be called `bh_da_sne`.

# Usage #


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

