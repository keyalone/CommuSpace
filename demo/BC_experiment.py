# %% [markdown]
# # CommuNiche analysis the human breast cancer micro-environment
#
# To facilitate accessibility and reproducibility of our results, we have deposited all datasets used in this study on [Zenodo](https://zenodo.org). Below we will demonstrate the applying of CommuNiche for the exploration of the human breast cancer data.
#
# **Workflow**
# 1. Load spatial transcriptomics data and LR human reference
# 2. Construction of communication tensor
# 3. Tensor decomposition
# 4. Niche identification
# 5. Niche interpretation

# ## Load datasets
#
# First, let’s load the package and the data. We 
# focus on human breast cancer data from Xenium ST technology (Replication 1), 
# The processed data are available at the [Zenodo data repository] (https://doi.org/10.5281/zenodo.4739739). 
# Aftet filtering low-quality genes and cells, a final set of 11920 genes and 306 spots for SRT data and 11920 
# genes and 3024 cells for scRNA-seq data are included.



