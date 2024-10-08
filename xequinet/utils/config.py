from typing import Union, List, Optional
from pydantic import BaseModel


class NetConfig(BaseModel):
    """
    Network configuration
    """  
    # non-essential
    run_name: str = "my_task"                      # name of the run

    # configurations about the model
    version: str = "xpainn"                        # model version
    embed_basis: str = "gfn2-xtb"                  # embedding basis type
    aux_basis: str = "aux56"                       # auxiliary basis type
    node_dim: int = 128                            # node irreps for the input
    edge_irreps: str = "128x0e + 64x1o + 32x2e"    # edge irreps for the input
    hidden_dim: int = 64                           # hidden dimension for the output
    hidden_irreps: str = "64x0e + 32x1o + 16x2e"   # hidden irreps for the output
    rbf_kernel: str = "bessel"                     # radial basis function type
    num_basis: int = 20                            # number of the radial basis functions
    cutoff: float = 5.0                            # cutoff distance for the neighbor atoms
    cutoff_fn: str = "cosine"                      # cutoff function type
    max_edges: int = 100                           # maximum number of the edges
    action_blocks: int = 3                         # number of the action blocks
    activation: str = "silu"                       # activation function type
    norm_type: str = "nonorm"                      # normalization layer type
    output_mode: str = "scalar"                    # task type (`scalar` is for energy like, `grad` is for force like, etc.)
    reduce_op: Optional[str] = "sum"               # reduce operations act on element output in graph  
    output_dim: int = 1                            # output dimension of multi-task (only for `scalar` mode)
    atom_ref: Optional[Union[str, dict]] = None    # atomic reference (only for `scalar` mode)
    batom_ref: Optional[Union[str, dict]] = None   # base atomic reference (only for `scalar` mode)
    node_average: Union[bool, float] = False       # whether to add the node average to the output (only for `scalar` mode)
    default_length_unit: str = "Angstrom"          # unit of the input coordinates
    default_property_unit: str = "eV"              # unit of the input properties
    default_dtype: str = "float32"                 # default data type
    # additional configurations for n-order tensor output 
    order: int = 2                                 # rank of output tensor
    required_symm: str = "ij"                      # indices symmetry of the tensor, "ij" for arbitary rank-2 tensor etc.
    vector_space: Optional[dict] = None            # vector space for the output tensor
    target_elem: Optional[Union[str, int]] = None  # for element-wise training of atomic shielding constant 
    mat_hidden_dim: int = 64                       # hidden dimension of each irrep feature in network
    max_l: int = 4                                 # maximum angular momentum required in network
    irreps_out: Optional[str] = None               # output layout required by user

    # configurations about the dataset
    dataset_type: str = "normal"                   # dataset type (`memory` is for the dataset in memory, `disk` is for the dataset on disk)
    data_root: Optional[str] = None                # root directory of the dataset
    data_files: Optional[Union[List[str], str]] = None  # list of the data files
    processed_name: Optional[str] = None           # name of the processed dataset
    mem_process: bool = True                       # whether to process the dataset in memory
    label_name: Optional[str] = None               # name of the label
    blabel_name: Optional[str] = None              # name of the basis label
    force_name: Optional[str] = None               # name of the force
    bforce_name: Optional[str] = None              # name of the basis force
    label_unit: Optional[str] = None               # unit of the input label
    blabel_unit: Optional[str] = None              # unit of the input base label
    force_unit: Optional[str] = None               # unit of the input force
    bforce_unit: Optional[str] = None              # unit of the input base force
    batch_size: int = 64                           # training batch size
    vbatch_size: int = 64                          # validation batch size

    # configurations about the training
    ckpt_file: Optional[str] = None                # checkpoint file to load
    resume: bool = False                           # whether to resume the training
    finetune: bool = False                         # whether to finetune the model
    warmup_scheduler: str = "linear"               # warmup scheduler type
    warmup_epochs: int = 10                        # number of the warmup epochs
    max_epochs: int = 300                          # maximum number of the training epochs
    max_lr: float = 5e-4                           # maximum learning rate
    min_lr: float = 0.0                            # minimum learning rate
    lossfn: str = "smoothl1"                       # loss function type
    force_weight: float = 0.99                     # weight of the force loss
    grad_clip: Optional[float] = None              # gradient clip
    optimizer: str = "adamW"                       # optimizer type
    optim_kwargs: dict = {}                        # kwargs for the optimizer
    lr_scheduler: str = "cosine_annealing"         # learning rate scheduler type
    lr_sche_kwargs: dict = {}                      # kwargs for the learning rate scheduler
    early_stop: Optional[int] = None               # number of the epochs to wait before stopping the training
    ema_decay: Optional[float] = None              # exponential moving average decay

    # configurations about the logging
    save_dir: str = './'                           # directory to save the model
    best_k: int = 1                                # number of the best models to keep
    log_file: str = "loss.log"                     # name of the logging file
    log_step: int = 50                             # number of the steps to log the training information
    log_epoch: int = 1                             # number of the epochs to log the information

    # others
    seed: Optional[int] = None                     # random seed
    num_workers: int = 0                           # number of the workers for the data loader

    def model_hyper_params(self):
        hyper_params = self.model_dump(include={
            "version", "embed_basis", "aux_basis", "node_dim", "edge_irreps", "hidden_dim", "hidden_irreps",
            "rbf_kernel", "num_basis", "cutoff", "cutoff_fn", "max_edges", "action_blocks",
            "activation", "norm_type", "output_mode", "output_dim", "order", "required_symm", 
            "max_l", "mat_hidden_dim",
            "atom_ref", "batom_ref", "node_average", "default_property_unit", "default_length_unit",
            "default_dtype",
        })
        return hyper_params


if __name__ == "__main__":
    config = NetConfig()
    print(config.model_hyper_params())
