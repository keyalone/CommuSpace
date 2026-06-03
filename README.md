<h1> <p align="center">
    Communication-space representation of multicellular tissue organization
</p> </h1>



**CommuSpace**,  a relational representation framework that embeds cells into a communication space defined by local intercellular communication architecture in spatial transcriptomics data. In this space, similarity reflects shared communication architecture rather than transcriptional identity or neighboring cellular composition alone, allowing transcriptionally distinct cells to converge toward shared organizational states while separating similar phenotypes across distinct communication contexts. Stable organizational states emerge within communication space as recurrent communication ecosystems.

<p align="center">
  <img src="Fig1.png" width="900"> 
</p>

### Installation

**CommuSpace** has been developed and tested on **Ubuntu 18.04.2 LTS and Window 11 x64**.

To install **CommuSpace**, please follow these steps:

1. **Set up a Conda environment for CommuSpace**:

    ```bash
    conda env create -f environment.yml
    conda activate CommuSpace
    ```

    In this step, we install all the requirments for CommuSpace


2. **Clone the repository and install CommuSpace**:

    ```bash
    git clone https://github.com/keyalone/CommuSpace.git
    cd CommuSpace
    pip install .
    ```

3. **Verify the installation in Python**:

    ```python
    import CommuSpace 
    ```

## Reproducibility
To facilitate usability and reproducibility of our results, we have uploaded all data used in this study to [Zenodo](https://zenodo.org/records/20519325) for public access.

We provide source codes for reproducing the CommuSpace analysis in the main text in the `demos` directory.

- [Communication ecosystems resolve tumor core architecture and specialized boundary states in human breast cancer](demo/BC_rep.ipynb)
- [Communication space reveals ecosystem-associated immune-state remodeling across tumor ecosystems](demo/BC_CD8.ipynb)


 
## Contact information
Please do not hesitate to contact Dr. Hui-Sheng Li (<lihs@mails.ccnu.edu.cn>) or Prof. Xiao-Fei Zhang (<zhangxf@ccnu.edu.cn>) to seek any clarifications regarding any contents.