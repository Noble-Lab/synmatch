# synmatch
## Linking cells across single-cell modalities by synergistic matching of neighborhood structure

Synmatch finds direct matching of cells between different measurements by exploiting information about neighborhood structure in each modality. It takes as input two matrices of single-cell profiles measuring different cellular properties, such as gene expression and chromatin accessibility, and outputs a matching of the cells across the datasets.


#### Dependencies
Synmatch is implemented in Python and uses Docker as well as the common numpy, sklearn, and scipy.
Note that Synmatch relies on Coopraiz, an ultra-fast software for submodular optimizations developed by Jeff Bilmes at [smr.ai](https://smr.ai/), which is included as a Docker container. Make sure you have Docker installed and running prior to running Synmatch.


#### How to run
python bin/synmatch.py data/example\_RNA.npy data/example\_ATAC.npy outputfile.txt


#### Questions
Feel free to email Borislav Hristov: borislav @ uw.edu 