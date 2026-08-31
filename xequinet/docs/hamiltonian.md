# XequiNet workflow for Machine Learning Kohn-Sham Hamiltonian
This document describes the workflow for training and inference of the Machine Learning Kohn-Sham Hamiltonian via models implemented in XequiNet. 

## Overview
The `hamiltonian` branch of XequiNet gets inspired by the code implementation of [`OrbNet-Equi`](https://zenodo.org/records/6568437) and [`QHNet`](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHNet) for data generation, model implementation and training workflow of ML Kohn-Sham Hamiltonian. Currently, supported models include `xQHNet`, `XPaiNN-Orb` and `XPaiNN-Orb/X2`, where `xQHNet` is a modified $E(3)$-equivariant version of `QHNet`, `XPaiNN-Orb` is a combination of `XPaiNN` and post-transformation modules of `QHNet`, and `XPaiNN-Orb/X2` is a further extension of `XPaiNN-Orb` with quantum chemistry informed features by the overlap integerals of atomic orbitals (the `X2` embedding). 

## Training 
### Dataset
See [training doc](./training.md) for the structure of the hdf5 dataset. For Hamiltonian training, make sure the dataset contains the following properties:
- `"<label_property>"`: `(M, Nbasis, Nbasis)`, `float64`. The label property of the Hamiltonian matrix in the atomic orbital basis or the orthonormal basis, where `M` is the number of configurations and `Nbasis` is the total number of basis functions for the molecule sample. The name of the property can be customized, but it should be consistent with the configuration file.
- `"<edge_attribute>"`: `(M, Nbasis, Nbasis)`, `float64`. The overlap matrix in the atomic orbital basis for training the `XPaiNN-Orb/X2` model, the matrix is used to generate the `X2` pair feature. Currently, the overlap matrix should be computed in the same basis set as the label Hamiltonian matrix. 
- `"<base_property>"`: `(M, Nbasis, Nbasis)`, `float64`. Baseline Hamiltonian matrix for $\Delta$-learning. 

An example of configuration file for dataset definition of a $\Delta$-learning `XPaiNN-Orb/X2` is like
```json
{
    "irreps_out": "3x0e + 2x1o + 1x2e",
    "target_basisname": "def2-svp",
    "m_idx_map_type": "pyscf",
    "possible_elements": ["H", "C", "O", "N", "F"],
    
    "default_property_unit": "Hartree",
    "default_length_unit": "Angstrom",
    "default_dtype": "float64",

    "dataset_type": "memory",
    "data_root": "/opt/XequiNet_tensor/dataset/QH9",
    "data_files": "QH9_stable.hdf5",
    "processed_name": "qh9_rc5.0_fp64",
    "label_name": "Fock",
    "blabel_name": "init_Fock",
    "edge_attr": "overlap",
    "label_unit": "Hartree",
    "blabel_unit": "Hartree",
}
```
where `irreps_out` is the output layout of atom-wise Hamiltonian matrix subblocks in the padding format (see [QHNet paper](https://arxiv.org/abs/2306.04922) for details), `target_basisname` is the basis set used for the label Hamiltonian matrix, `m_idx_map_type` is the mapping indices between the atomic orbital indices within atomic sub-shell and e3nn Irreps layout, which can differ among different software output, like PySCF and ORCA.

### Model Definition
A typical configuration file for defining the `XPaiNN/Orb` model is like 
```json
{
    "version": "xpainn-mat",

    "embed_basis": "gfn2-xtb",
    "aux_basis": "aux56",
    "node_dim": 128,
    "edge_irreps": "128x0e+128x1o+128x2e+128x3o+128x4e",
    "rbf_kernel": "bessel",
    "num_basis": 20,
    "cutoff": 5.0,
    "cutoff_fn": "exponential",
    "action_blocks": 3,
    "activation": "silu",
    
    "num_mat_conv": 1,
    "max_l": 4,
    "mat_hidden_dim": 64,
    "mat_block_dim": 32,
    "pair_rbf_kernel": "expbern",
    "pair_num_basis": 32,
    "pair_cutoff": 12.0,
}
```
Use `version` to specify the model type. The model definition can be seperated into two parts, the first part is the `XPaiNN` message passing blocks and the second is the post-transformation blocks for building the matrix. In the latter, `num_mat_conv` is the number of the transformation blocks, which should be smaller than or equal to `action_blocks`, the depth of the message passing block. `max_l` is the maximum angular momentum of hidden $E(3)$-equivariant features with `mat_hidden_dim` as the feature channel dimension. Keywords with `pair` prefix are used for the RBF kernel for generating the pair features in the complete graph. 

Defining `XPaiNN-Orb/X2` is similar, major differences are in the post-transformation blocks, see the following example configuration file:
```json
{
    "version": "xpainn3-mat",

    "embed_basis": "gfn2-xtb",
    "aux_basis": "aux56",
    "node_dim": 128,
    "edge_irreps": "128x0e+128x1o+128x2e+128x3o+128x4e",
    "rbf_kernel": "bessel",
    "num_basis": 20,
    "cutoff": 5.0,
    "cutoff_fn": "cosine",
    "action_blocks": 3,
    "activation": "silu",
    
    "num_mat_conv": 1,
    "max_l": 4,
    "mat_hidden_dim": 64,
    "mat_block_dim": 32,
    "edge_attr": "overlap",
    "pair_num_basis": 36,
}
```
where the RBF kernel is replaced with the `X2` pair embeddings. Here `pair_num_basis` is pre-defined by the `irreps_out` field in the dataset part. For example, the `36` is evaluated by the degeneracies of irreps in `3x0e + 2x1o + 1x2e`, which is $(3+2+1)^2 = 36$.

### Training with a certain regularization loss
The `lossfn` selects the type of Loss function for refining the matrix elements, where `matloss` is the joint loss function of MAE and RMSE. For additional regularization term, use `output_mode` and other related keywords to specify, for example, set `output_mode` to `orbital_grad` to add the DIIS-like orbital gradient regularization term, set it to `wavefunction` to specify the wavefunction regularization which also contains the DIIS-like term. Use `reg_weight` to control the weight of the regularization term. For the DIIS-like term, make sure the label Hamiltonian matrix is in the orthonormal basis and set `ortho_transform` to `True` in the configuration file.

## Inference
Use `xequinet/run/fock_predict.py` to run the inference of the trained model. Use the following command
```bash
python /path/to/xequinet/run/fock_predict.py \
    -c $model_checkpoint \
    --xc $xc_functional \
    --transform \
    input_xyz_file
```
The script takes the trained model checkpoint, the xc functional name and the input xyz file as arguments. The `--transform` flag is used to specify whether the output Fock matrix is in the orthonormal basis or not. The prediction results will be saved under the current directory as PySCF checkpoint files (*.chk) per molecule, which can be used for further post-processing or analysis, via PySCF or other quantum chemistry software via MOKIT.