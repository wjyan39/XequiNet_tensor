## XequiNet
XequiNet is a package implemented for property prediction of chemical molecules or periodical systems with equivariant graph neural network.

For the original repository, see <https://github.com/X1X1010/XequiNet.git>.

## Requirements
**The following versions of these packages are only recommended for use.**

python 3.9<br>
pytorch 2.0<br>
pyg (follow pytorch)<br>
pytorch-cluster (follow pytorch)<br>
pytorch-scatter (follow pytorch)<br>
e3nn 0.5<br>
pytorch-warmup 0.1<br>
pydantic 2.6<br>
ase 3.22<br>
pyscf 2.4

## Installation of dependencies
### For GLIBC <= 2.27
```
conda create -n <env_name> python=3.9 numpy=1.26 scipy h5py 
source activate <env_name>
conda install pytorch=2.0.1 pytorch-cuda==11.7 -c pytorch -c nvidia
conda install pyg=2.3 -c pyg 
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install pydantic==2.6
pip install tqdm pyscf==2.4 e3nn==0.5.1 pytorch-warmup
pip install ase==3.22
conda deactivate
```

### For latest version of packages
```
conda create -n <env_name> python=3.11 numpy scipy h5py -c conda-forge
source activate <env_name>
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu126.html
pip install pydantic==2.6
pip install tqdm pyscf e3nn pytorch-warmup
pip install ase
conda deactivate
```

### Extra requirements

**The geomeTRIC package is used for geometry optimization.**

geometric 1.0

**The TBLite packages are used for delta learning with GFN2-xTB as base.**

tblite 0.3<br>
tblite-python 0.3

**MOKIT is used for transforming wavefunction files from model predicted KS Hamiltonian.**

mokit `conda install -c conda-forge -c mokit mokit`

## Setups
### From Source
Once the requirements are installed, running
```
pip install -e .
```

## Usage
See the markdown files in `docs` for details.

`docs/training.md`: Training and testing with dataset in `hdf5` format. No support for CPU training at this time.

`docs/inference.md`: Prediction with a trained model `xxx.pt`.

`docs/geometry.md`: Geometry optimization and molecular dynamics with a **JIT** model `xxx.jit`.