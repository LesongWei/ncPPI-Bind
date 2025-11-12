 # **Adaptive Geometric Deep Learning improves Non-Canonical Peptide–Protein Binding Site Prediction**
 
 ncPPI-Bind is an adaptive geometric deep learning framework designed to predict binding sites between proteins and peptides containing non-canonical amino acids (ncAAs).
## 1 Installation
### 1.1 Create conda environment
        
        conda create -n pytorch python=3.10
        conda activate pytorch

### 1.2 Requirements
We recommend installing the environment using the provided environment.yaml file to ensure compatibility:\
        
        conda env update -f environment.yaml --prune

If this approach fails or Conda is not available, you can manually install the main dependencies as listed below:
        
        python  3.10
        biopython 1.81
        tokenizers 0.13.3
        torch 2.2.2+cu118
        torchaudio 2.2.2+cu118
        torch-cluster 1.6.3+pt22cpu
        torch-geometric 2.7.0
        torch-scatter 2.1.2+pt22cpu
        torch-sparse 0.6.18+pt22cpu
        torchmetrics 1.8.2
        torchvision 0.17.2+cu118
        transformers 4.30.2

## 2 Usage
### Training
        
        python train.py 





        





