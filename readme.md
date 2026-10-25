# HieraPTM and HieraFuncPhos

This repository provides the model implementations and dataset splits for HieraPTM and HieraFuncPhos, supporting protein post-translational modification site prediction and functional phosphorylation site identification.

The project addresses two related prediction tasks:

- **Post-translational modification site prediction**: predicting whether candidate residues in a protein are likely to undergo a specific type of post-translational modification.
- **Functional phosphorylation site prediction**: assessing the functional relevance of phosphorylation sites to identify candidates for subsequent biological studies and experimental validation.

## Models

### HieraPTM

HieraPTM is a hierarchical prediction framework that integrates protein sequence and three-dimensional structural information. It combines local sequence patterns, protein language model representations, and spatial relationships between residues to predict post-translational modification sites.

The project covers eight site prediction tasks involving acetylation, glycosylation, methylation, phosphorylation, SUMOylation, and ubiquitination.

### HieraFuncPhos

HieraFuncPhos builds on the pretrained representations of HieraPTM and further integrates protein-level semantic information and structural dynamics for functional phosphorylation site prediction.

The model considers both the local environment of a site and its dynamic relationships with distant residues, extending modification site identification to the assessment of functional relevance.

## Project Structure

- `HieraPTM/code/`: HieraPTM model definitions, training and evaluation entry points, and configuration files.
- `HieraPTM/data/splits/`: dataset splits for the eight post-translational modification prediction tasks.
- `HieraFuncPhos/code/`: HieraFuncPhos model definitions, training and evaluation entry points, and configuration files.
- `HieraFuncPhos/data splits/`: training, validation, and test splits for functional phosphorylation prediction across rep01–rep09.

## Python Environment

**Python 3.11** is recommended. Create an isolated environment using Conda:

```bash
conda create -n hieraptm python=3.11 -y
conda activate hieraptm
```

The main dependencies for the two models are listed below:

| Dependency | Version requirement |
|---|---|
| PyTorch | >=2.5, <3 |
| NumPy | >=1.24, <3 |
| pandas | >=2.0, <3 |
| scikit-learn | >=1.3, <2 |
| Matplotlib | >=3.7, <4 |
| psutil | >=5.9, <8 |
| PyArrow | >=14 |

The HieraPTM training workflow uses CUDA. First, install an appropriate CUDA-enabled PyTorch build for your operating system and GPU driver using the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

Then install the dependencies for both models from the repository root:

```bash
python -m pip install -r HieraPTM/code/requirements.txt -r HieraFuncPhos/code/requirements.txt
```

## Usage

The code for each model is organized in its respective directory and includes training, validation, and testing workflows. For detailed instructions, please refer to the README in the corresponding code directory.

## Citation

If this project is helpful to your research, please cite the associated paper. Full citation details will be added in a future update.
