conda create -y -n xequiten python=3.9 numpy scipy h5py
source activate xequiten
conda install pytorch==2.0.1 pytorch-cuda==11.7 -c pytorch -c nvidia
conda install pyg=2.3 -c pyg
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install pydantic==2.6
pip install tqdm pyscf==2.4 e3nn==0.5.1 pytorch-warmup
pip install ase==3.22
conda deactivate 
