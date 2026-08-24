from typing import Tuple, List, Optional
import os
import heapq

import numpy as np 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
# from torch_scatter import scatter

from torch.optim.swa_utils import AveragedModel
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .functional import (
    resolve_lossfn,
    resolve_optimizer,
    resolve_lr_scheduler,
    resolve_warmup_scheduler,
)
from .config import NetConfig
from .logger import ZeroLogger
from .qc import get_default_unit, ELEMENTS_DICT
from .qc_matrice_graph import QCMatriceBuilder, OrbitalCalculator


class loss2file:
    def __init__(self, loss: float, ptfile: str, epoch: int):
        self.loss = loss
        self.ptfile = ptfile
        self.epoch = epoch

    def __lt__(self, other: "loss2file"):
        # overloading __lt__ inversely to realize max heap
        return self.loss > other.loss


class AverageMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((1,), device=self.device)
        self.cnt = torch.zeros((1,), dtype=torch.int32, device=self.device)
    
    def update(self, val: float, n: int = 1):
        self.sum += val
        self.cnt += n
    
    def reduce(self) -> float:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg.item()


class EarlyStopping:
    def __init__(
        self, patience: int = None, min_delta: float = 0.0, min_lr: float = 1e-6,
    ):
        self.patience = patience if patience is not None else float("inf")
        self.min_delta = min_delta
        self.min_lr = 1e-6 if min_lr == 0.0 else min_lr
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, best_loss: float, lr: float):
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        elif lr <= self.min_lr:
            self.stop = True
        else:
            self.counter = 0
        return self.stop


class Trainer:
    """
    General trainer class for training neural networks.
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler],
        log: ZeroLogger,
    ):
        """
        Args:
            `model`: DistributedDataParallel model
            `config`: network configuration
            `device`: torch device
            `train_loader`: training data loader
            `valid_loader`: validation data loader
            `dist_sampler`: distributed sampler
            `log`: logger
        """
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dist_sampler = dist_sampler
        self.log = log

        # set loss function
        self.lossfn = resolve_lossfn(config.lossfn).to(device)
        # set optimizer
        self.optimizer = resolve_optimizer(
            optim_type=config.optimizer,
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.max_lr,
            **config.optim_kwargs,
        )
        # set lr scheduler
        self.lr_scheduler = resolve_lr_scheduler(
            sched_type=config.lr_scheduler,
            optimizer=self.optimizer,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            max_epochs=config.max_epochs,
            steps_per_epoch=len(train_loader),
            **config.lr_sche_kwargs,
        )
        # set warmup scheduler
        if config.lr_scheduler == "plateau":
            warm_steps = config.warmup_epochs
        else:
            warm_steps = config.warmup_epochs * len(train_loader)
        self.warmup_scheduler = resolve_warmup_scheduler(
            warm_type=config.warmup_scheduler,
            optimizer=self.optimizer,
            warm_steps=warm_steps,
        )
        # set early stopping (only work when lr_scheduler is plateau)
        self.early_stop = EarlyStopping(
            patience=config.early_stop, min_lr=config.min_lr
        )
        # exponential moving average
        self.ema_model = None
        if config.ema_decay is not None:
            ema_model = AveragedModel(
                self.model.module,
                avg_fn=lambda avg_param, param, num_avg: \
                    config.ema_decay * avg_param + (1 - config.ema_decay) * param,
                device=device,
            )
            self.ema_model = ema_model
        # loss recording, model saving and logging
        self.meter = AverageMeter(device=device)
        self.best_l2fs: List[loss2file] = [
            loss2file(float("inf"), os.path.join(config.save_dir, f"{config.run_name}_{i}.pt"), 0)
            for i in range(config.best_k)
        ]  # a max-heap, actually it is a min-heap
        
        # load checkpoint
        self.start_epoch = 1
        if config.ckpt_file is not None:
            self._load_params(config.ckpt_file)


    def _load_params(self, ckpt_file: str):
        state = torch.load(ckpt_file, map_location=self.device)
        self.model.module.load_state_dict(state["model"], strict=False)
        if self.config.resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.warmup_scheduler.load_state_dict(state["warmup_scheduler"])
            self.start_epoch = state["epoch"] + 1
            for l2f in self.best_l2fs:
                if os.path.isfile(l2f.ptfile):
                    pt_state = torch.load(l2f.ptfile, map_location=self.device)
                    l2f.loss = pt_state["loss"] if "loss" in pt_state else float("inf")
        self.log.f.info(f" --- Loaded checkpoint from {ckpt_file}")


    def _save_params(self, model: nn.Module, ckpt_file: str, loss: float = None):
        state = {
            "model": model.state_dict(),
            "epoch": self.epoch,
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict(),
            "config": self.config.model_hyper_params(),
        }
        torch.save(state, ckpt_file)


    def save_best_k(self, model: nn.Module, curr_loss: float):
        if curr_loss < self.best_l2fs[0].loss:
            l2f = heapq.heappop(self.best_l2fs)
            l2f.loss = curr_loss
            l2f.epoch = self.epoch
            self._save_params(model, l2f.ptfile, l2f.loss)
            heapq.heappush(self.best_l2fs, l2f)


    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            # forward propagation
            pred = self.model(data)
            real = data.y - data.base_y if hasattr(data, "base_y") else data.y
            loss = self.lossfn(pred, real)
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            with torch.no_grad():
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train MAE: {mae:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae=mae,
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                pred = self.model(data)
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
        mae = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(f"Validation MAE: {mae:10.7f}")
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(mae, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.model.module, mae)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                pred = self.ema_model(data)
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss.item(), real.numel())
        mae = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info("EMA Valid MAE: {mae:10.7f}".format(mae=mae))
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(mae, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.ema_model.module, mae)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")

        # training loop
        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            self.train1epoch()
            if self.ema_model is None:
                self.validate()
            else:
                self.ema_validate()
            if self.early_stop.stop:
                self.log.f.info(f" --- Early Stopping at Epoch {iepoch}")
                break
        
        self.log.f.info(" --- Training Completed")
        self.log.f.info(f" --- Best Valid MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile} at Epoch {self.best_l2fs[-1].epoch}")


class WithForceMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((2,), device=self.device)
        self.cnt = torch.zeros((2,), dtype=torch.int32, device=self.device)

    def update(self, energy: float, force: float, n_ene: int, n_frc: int):
        self.sum[0] += energy; self.sum[1] += force
        self.cnt[0] += n_ene; self.cnt[1] += n_frc

    def reduce(self) -> Tuple[float, float]:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg[0].item(), avg[1].item()
    

class GradTrainer(Trainer):
    """
    Trainer class for scalar and relative gradient property
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: DistributedSampler,
        log: ZeroLogger,
    ):
        """
        Args:
            `model`: DistributedDataParallel model
            `config`: network configuration
            `device`: torch device
            `train_loader`: training data loader
            `valid_loader`: validation data loader
            `dist_sampler`: distributed sampler
            `log`: logger
        """
        super().__init__(
            model, config, device, train_loader, valid_loader, dist_sampler, log
        )
        assert config.force_weight <= 1.0
        self.meter = WithForceMeter(self.device)

    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            # forward propagation
            data.pos.requires_grad_(True)
            predE, predF = self.model(data)
            realE, realF = data.y, data.force
            if hasattr(data, "base_y") and hasattr(data, "base_force"):
                realE -= data.base_y
                realF -= data.base_force
            lossE = self.lossfn(predE, realE)
            lossF = self.lossfn(predF, realF)
            loss = (1 - self.config.force_weight) * lossE + self.config.force_weight * lossF
            # backward propagation
            self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            with torch.no_grad():
                l1lossE = F.l1_loss(predE, realE, reduction="sum")
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                maeE, maeF = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:4d}/{nstep}]   lr: {lr:3e}   train MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        maeE=maeE,
                        maeF=maeF,
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        for data in self.valid_loader:
            data = data.to(self.device)
            data.pos.requires_grad_(True)
            predE, predF = self.model(data)
            with torch.no_grad():
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(predE, realE, reduction="sum") 
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
        maeE, maeF = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(f"Validation MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}")
        mae = (1 - self.config.force_weight) * maeE + self.config.force_weight * maeF
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(mae, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.model.module, mae)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


    
    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        for data in self.valid_loader:
            data = data.to(self.device)
            data.pos.requires_grad_(True)
            predE, predF = self.ema_model(data)
            with torch.no_grad():
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(predE, realE, reduction="sum") 
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
        maeE, maeF = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(f"EMA Validation MAE: Energy {maeE:10.7f}  Force {maeF:10.7f}")
        mae = (1 - self.config.force_weight) * maeE + self.config.force_weight * maeF
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(mae)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(mae, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.ema_model.module, mae)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


class CSCTrainer(Trainer):
    """
    Trainer class for Chemical Shielding Tensors/Constants as well as Shifts.
    """
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        super().__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
        if config.target_elem is not None:
            if isinstance(config.target_elem[0], str):
                self.target_elem = [ELEMENTS_DICT[ele] for ele in config.target_elem]
            else:
                self.target_elem = config.target_elem
        else:
            self.target_elem = None
        self.trace_out = config.output_dim == 1 
        # self.lossfn = resolve_lossfn(config.lossfn, reduction="none")
    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch) 
        
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device) 
            # generate the node mask 
            if hasattr(data, "label_mask"):
                batch_label_mask = data.label_mask
            else:
                batch_label_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=self.device)
            if self.target_elem is not None:
                batch_elem_mask = torch.zeros(data.y.shape[0], dtype=torch.bool, device=self.device)
                for ele in self.target_elem:
                    batch_elem_mask = torch.logical_or(batch_elem_mask, data.at_no == ele)
                batch_node_mask = torch.logical_and(batch_label_mask, batch_elem_mask) 
            else:
                batch_node_mask = batch_label_mask
            # forward propagation
            res = self.model(data)
            # mask the results 
            non_label = torch.all(batch_node_mask == False)
            all_label = torch.all(batch_node_mask == True)
            if non_label:
                real = res.detach().clone()
                pred = res 
            elif all_label:
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y 
                pred = res 
            else:
                if self.trace_out: 
                    pred = res[batch_node_mask]
                    real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                    real = real[batch_node_mask]
                else:
                    pred = res[batch_node_mask, ...]
                    real = data.y[batch_node_mask, ...]
            loss = self.lossfn(pred, real)
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record train loss
            with torch.no_grad():
                # l1loss = F.l1_loss(pred, real, reduction="sum")
                batch_loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(batch_loss.item(), real.numel())
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train Loss: {mae:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae=mae,
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.model(data)
                if hasattr(data, "label_mask"):
                    batch_label_mask = data.label_mask
                else:
                    batch_label_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=self.device)
                if self.target_elem is not None:
                    batch_elem_mask = torch.zeros(data.y.shape[0], dtype=torch.bool, device=self.device)
                    for ele in self.target_elem:
                        batch_elem_mask = torch.logical_or(batch_elem_mask, data.at_no == ele)
                    batch_node_mask = torch.logical_and(batch_label_mask, batch_elem_mask)  
                else:
                    batch_node_mask = batch_label_mask
                if self.trace_out: 
                    pred = res[batch_node_mask]
                    real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                    real = real[batch_node_mask]
                else:
                    pred = res[batch_node_mask, ...]
                    real = data.y[batch_node_mask, ...]
                batch_loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(batch_loss.item(), real.numel())
        deviation = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(f"Validation Loss: {deviation:10.7f}")
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(deviation)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(deviation, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.model.module, deviation)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")

    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.ema_model(data)
                if hasattr(data, "label_mask"):
                    batch_label_mask = data.label_mask
                else:
                    batch_label_mask = torch.ones(data.y.shape[0], dtype=torch.bool, device=self.device)
                if self.target_elem is not None:
                    batch_elem_mask = torch.zeros(data.y.shape[0], dtype=torch.bool, device=self.device)
                    for ele in self.target_elem:
                        batch_elem_mask = torch.logical_or(batch_elem_mask, data.at_no == ele)
                    batch_node_mask = torch.logical_and(batch_label_mask, batch_elem_mask)  
                else:
                    batch_node_mask = batch_label_mask
                if self.trace_out: 
                    pred = res[batch_node_mask]
                    real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                    real = real[batch_node_mask]
                else:
                    pred = res[batch_node_mask, ...]
                    real = data.y[batch_node_mask, ...]
                batch_loss = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(batch_loss.item(), real.numel())
        deviation = self.meter.reduce()
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info("EMA Valid Loss: {err:10.7f}".format(err=deviation))
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(deviation)
            lr = self.optimizer.param_groups[0]["lr"]
            self.early_stop(deviation, self.best_l2fs[0].loss, lr)
        self.save_best_k(self.ema_model.module, deviation)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


class GraphMeter:
    """
    Meteric class for evaluating error metrics containing both node and edge labels.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.accum_loss = torch.zeros((3,), device=self.device)
        self.counter = torch.zeros((3,), device=self.device, dtype=torch.int32) 

    def update(self, node_datum: float, edge_datum: float, total_datum, num_node: int, num_edge: int, num_tot):
        self.accum_loss[0] += node_datum; self.accum_loss[1] += edge_datum; self.accum_loss[2] += total_datum
        self.counter[0] += num_node; self.counter[1] += num_edge; self.counter[2] += num_tot
    
    def reduce(self) -> Tuple[float, float, float]:
        this_accum_loss = self.accum_loss.clone() 
        this_counter = self.counter.clone() 
        dist.all_reduce(this_accum_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(this_counter, op=dist.ReduceOp.SUM) 
        this_avg = this_accum_loss / this_counter 
        return this_avg[0].item(), this_avg[1].item(), this_avg[2].item()


class QCMatTrainer(Trainer):
    """
    Trainer class for general matrice properties calculated from quantum chemistry method
    with a given basis set layout.
    """
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        super().__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
        self.meter = GraphMeter(self.device) 
        self.node_weight = config.reg_weight if config.reg_weight > 0.0 else 1.0 
    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch) 
        
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device) 
            # forward propagation
            res = self.model(data)
            pred_pad_node, pred_pad_edge = res[0], res[1]
            real_pad_node = data.node_label - data.node_base if hasattr(data, 'node_base') else data.node_label
            real_pad_edge = data.edge_label - data.edge_base if hasattr(data, 'edge_base') else data.edge_label
            batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
            pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
            real_node, real_edge = real_pad_node[batch_mask_node], real_pad_edge[batch_mask_edge]
            batch_pred = torch.cat([pred_node, pred_edge], dim=0)
            batch_real = torch.cat([real_node, real_edge], dim=0)
            # loss = self.lossfn(batch_pred, batch_real)
            loss = self.node_weight * self.lossfn(pred_node, real_node) + self.lossfn(pred_edge, real_edge)
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # record l1 loss
            with torch.no_grad():
                l1loss_node = F.l1_loss(pred_node, real_node, reduction="sum")
                l1loss_edge = F.l1_loss(pred_edge, real_edge, reduction="sum")
                l1loss_total = F.l1_loss(batch_pred, batch_real, reduction="sum")
                self.meter.update(l1loss_node.item(), l1loss_edge.item(), l1loss_total.item(), real_node.size(0), real_edge.size(0), batch_real.size(0))
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train MAE: node: {mae_n:10.7f},  edge: {mae_e:10.7f},  total: {mae_tot:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae_n=mae[0],
                        mae_e=mae[1],
                        mae_tot=mae[2],
                    )
                )
        
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
                real_node = data.node_label[batch_mask_node] - data.node_base[batch_mask_node] if hasattr(data, 'node_base') else data.node_label[batch_mask_node]
                real_edge = data.edge_label[batch_mask_edge] - data.edge_base[batch_mask_edge] if hasattr(data, 'edge_base') else data.edge_label[batch_mask_edge]
                batch_pred = torch.cat([pred_node, pred_edge], dim=0)
                batch_real = torch.cat([real_node, real_edge], dim=0)
                node_l1loss = F.l1_loss(pred_node, real_node, reduction="sum")
                edge_l1loss = F.l1_loss(pred_edge, real_edge, reduction="sum")
                total_l1loss = F.l1_loss(batch_pred, batch_real, reduction="sum")
                self.meter.update(node_l1loss.item(), edge_l1loss.item(), total_l1loss.item(), real_node.size(0), real_edge.size(0), batch_real.size(0))
        mae = self.meter.reduce()
        self.log.f.info(f"Validation MAE: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.model.module, total_mae)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt") 

    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.ema_model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
                real_node = data.node_label[batch_mask_node] - data.node_base[batch_mask_node] if hasattr(data, 'node_base') else data.node_label[batch_mask_node]
                real_edge = data.edge_label[batch_mask_edge] - data.edge_base[batch_mask_edge] if hasattr(data, 'edge_base') else data.edge_label[batch_mask_edge]
                batch_pred = torch.cat([pred_node, pred_edge], dim=0)
                batch_real = torch.cat([real_node, real_edge], dim=0)
                node_l1loss = F.l1_loss(pred_node, real_node, reduction="sum")
                edge_l1loss = F.l1_loss(pred_edge, real_edge, reduction="sum")
                total_l1loss = F.l1_loss(batch_pred, batch_real, reduction="sum")
                self.meter.update(node_l1loss.item(), edge_l1loss.item(), total_l1loss.item(), real_node.size(0), real_edge.size(0), batch_real.size(0))
        mae = self.meter.reduce()
        self.log.f.info(f"EMA Validation MAE: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.ema_model.module, total_mae)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


class OrbGradTrainer(QCMatTrainer):
    """
    Trainer class additionally add Orbital Gradient as a regularization term.
    """
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        super(OrbGradTrainer, self).__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
        self._set_init(config, device)
    
    def _set_init(self, config:NetConfig, device:torch.device):
        self.mat_builder = QCMatriceBuilder(config.irreps_out, config.possible_elements, config.target_basisname)
        self.orbital_calculator = OrbitalCalculator(config.target_basisname, config.default_length_unit, config.ortho_transform)
        self.reg_weight:float = config.reg_weight if config.reg_weight > 0.0 else 1.0
        self.mat_builder.to(device)
        self._default_dtype = torch.float64 if config.default_dtype == "float64" else torch.float32 
        self._device = device
        if config.output_mode == "orbital":
            self.full_eigen_space = False 
        elif config.output_mode == "eigen":
            self.full_eigen_space = True

    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch) 
        
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device) 
            # forward propagation
            res = self.model(data)
            pred_pad_node, pred_pad_edge = res[0], res[1]
            real_pad_node, real_pad_edge = data.node_label, data.edge_label
            batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
            if hasattr(data, 'node_base'):
                pred_pad_node = pred_pad_node + data.node_base
            if hasattr(data, 'edge_base'):
                pred_pad_edge = pred_pad_edge + data.edge_base
            pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
            real_node, real_edge = real_pad_node[batch_mask_node], real_pad_edge[batch_mask_edge]
            batch_pred = torch.cat([pred_node, pred_edge], dim=0)
            batch_real = torch.cat([real_node, real_edge], dim=0)
            # calculate the regularization term  
            real_fock = self.mat_builder(real_pad_node, real_pad_edge, data.at_no, data.fc_edge_index)
            real_fock = real_fock.cpu().numpy().astype(np.float64)
            at_no = data.at_no.cpu().numpy()
            coords = data.pos.cpu().numpy().astype(np.float64)
            charge = data.charge.to(torch.long).item()
            ## orbital coefficients 
            orb_coeffs, nocc, _ = self.orbital_calculator(real_fock, at_no, coords, charge)
            ## fock ao 
            pred_fock = self.mat_builder(pred_pad_node, pred_pad_edge, data.at_no, data.fc_edge_index)
            real_fock = torch.from_numpy(real_fock).to(self._device).to(self._default_dtype)
            if not self.full_eigen_space: 
                ### occupied space + orbital grad
                ### i.e. Foo + Fvo = C_occ.T @ \delta Fao @ C_occ + C_virt.T @ \delta Fao @ C_occ = C.T @ Fao @ C_occ
                orb_ket = torch.from_numpy(orb_coeffs).to(self._device).to(self._default_dtype)  # all
                orb_bra = torch.from_numpy(orb_coeffs[:, :nocc]).to(self._device).to(self._default_dtype).T  # occ
                ## orbital gradient Fvo = C_virt.T @ Fao @ C_occ 
                orb_grad = orb_bra @ (pred_fock - real_fock) @ orb_ket
                reg_loss = torch.norm(orb_grad, p="fro")
            else:
                orb_ket = torch.from_numpy(orb_coeffs).to(self._device).to(self._default_dtype)            # all 
                orb_bra = orb_ket.T 
                ## wavefunction alignment loss 
                wa_error = orb_bra @ (pred_fock - real_fock) @ orb_ket
                reg_loss = torch.norm(wa_error, p="fro")
            loss = self.lossfn(batch_pred, batch_real) + self.reg_weight * reg_loss
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # for matrice, record the Frobenious norm distance 
            with torch.no_grad():
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train Error: node: {mae_n:10.7f},  edge: {mae_e:10.7f},  total: {mae_tot:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae_n=mae[0],
                        mae_e=mae[1],
                        mae_tot=mae[2],
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                real_pad_node = data.node_label - data.node_base if hasattr(data, 'node_base') else data.node_label
                real_pad_edge = data.edge_label - data.edge_base if hasattr(data, 'edge_base') else data.edge_label
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
        mae = self.meter.reduce()
        self.log.f.info(f"EMA Validation Error: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.model.module, total_mae)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt") 

    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.ema_model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                real_pad_node = data.node_label - data.node_base if hasattr(data, 'node_base') else data.node_label
                real_pad_edge = data.edge_label - data.edge_base if hasattr(data, 'edge_base') else data.edge_label
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
        mae = self.meter.reduce()
        self.log.f.info(f"EMA Validation Error: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.ema_model.module, total_mae)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


class DIISTrainer(OrbGradTrainer):
    """
    Trainer class additionally add DIIS error as a regularization term.
    """
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        assert config.ortho_transform == True
        super().__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch) 
        
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device) 
            # forward propagation
            res = self.model(data)
            pred_pad_node, pred_pad_edge = res[0], res[1]
            real_pad_node, real_pad_edge = data.node_label, data.edge_label
            batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
            pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
            real_node, real_edge = real_pad_node[batch_mask_node], real_pad_edge[batch_mask_edge]
            batch_pred = torch.cat([pred_node, pred_edge], dim=0)
            batch_real = torch.cat([real_node, real_edge], dim=0)
            # calculate the regularization term 
            real_fock = self.mat_builder(real_pad_node, real_pad_edge, data.at_no, data.fc_edge_index)
            pred_fock = self.mat_builder(pred_pad_node, pred_pad_edge, data.at_no, data.fc_edge_index)
            ## calculate the density matrix
            # real_fock = real_fock.cpu().numpy().astype(np.float64)
            cur_fock = real_fock.detach().cpu().numpy().astype(np.float64)
            at_no = data.at_no.cpu().numpy()
            coords = data.pos.cpu().numpy().astype(np.float64)
            charge = data.charge.to(torch.long).item()
            ## D = C_occ @ C_occ.T
            orb_coeffs, nocc, _ = self.orbital_calculator(cur_fock, at_no, coords, charge)
            cur_den = orb_coeffs[:, :nocc] @ orb_coeffs[:, :nocc].T
            cur_den = torch.from_numpy(cur_den).to(self._device).to(self._default_dtype)
            diis_error = cur_den @ pred_fock - pred_fock @ cur_den
            diis_loss = self.reg_weight * torch.norm(diis_error, p="fro")
            # calculate the loss
            loss = self.lossfn(batch_pred, batch_real) + diis_loss
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # for matrice, record the Frobenious norm distance 
            with torch.no_grad():
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train Error: node: {mae_n:10.7f},  edge: {mae_e:10.7f},  total: {mae_tot:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae_n=mae[0],
                        mae_e=mae[1],
                        mae_tot=mae[2],
                    )
                )


class OrbitalTrainer(QCMatTrainer):
    """
    Trainer class and template that additionally introduce SCF related regularization terms when training Fock matrices.
    Here, Wavefunction Alignment loss and Orbital Gradient loss are implemented.
    """
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        super(OrbitalTrainer, self).__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
        self._set_init(config, device)
    
    def _set_init(self, config:NetConfig, device:torch.device):
        # modules for building Fock matrices and calculating orbitals
        self.mat_builder = QCMatriceBuilder(config.irreps_out, config.possible_elements, config.target_basisname)
        self.orbital_calculator = OrbitalCalculator(config.target_basisname, config.default_length_unit, config.ortho_transform)
        self.mat_builder.to(device)
        # loss regularization parameters
        self.reg_weight:float = config.reg_weight if config.reg_weight > 0.0 else 1.0
        # assert config.wa_type in [0, 1, 2], f"Unsupported wavefunction alignment type: {config.wa_type}"
        self.wa_type:int = config.wa_type 
        self.num_states:int = config.num_states if config.num_states >= 0 else 1
        # data type and device
        self._default_dtype = torch.float64 if config.default_dtype == "float64" else torch.float32 
        self._device = device

    def _get_reg_term(self, data, real_fock, pred_fock):
        # calculate the regularization term
        ## orbital coefficients 
        real_fock = real_fock.cpu().numpy().astype(np.float64)
        at_no = data.at_no.cpu().numpy()
        coords = data.pos.cpu().numpy().astype(np.float64)
        charge = data.charge.to(torch.long).item()
        orb_coeffs, nocc, nvirt = self.orbital_calculator(real_fock, at_no, coords, charge)
        real_fock = torch.from_numpy(real_fock).to(self._device).to(self._default_dtype)
        ## type 0 and type 1 (num_states = 1) are WALoss from Liu et.al. ICLR 2025
        if self.wa_type == 0:
            orb_space = torch.from_numpy(orb_coeffs).to(self._device).to(self._default_dtype)
            wa_error = orb_space.T @ (pred_fock - real_fock) @ orb_space
            reg_loss = self.reg_weight * torch.norm(wa_error, p="fro")
        elif self.wa_type == 1:
            # wavefunction alignment loss
            num_states = min(self.num_states, nvirt)
            ## assume the valence (active) space is num_occ + num_states (the first n virtual orbital)
            orb_space_val = torch.from_numpy(orb_coeffs[:, :nocc+num_states]).to(self._device).to(self._default_dtype)
            wa_error = orb_space_val.T @ (pred_fock - real_fock) @ orb_space_val
            wa_loss_I = self.reg_weight * torch.norm(wa_error, p="fro")
            if self.num_states < nvirt:
                orb_space_virt = torch.from_numpy(orb_coeffs[:, nocc+num_states:]).to(self._device).to(self._default_dtype)
                orb_error_virt = orb_space_virt.T @ (pred_fock - real_fock) @ orb_space_virt
                # according to Liu et.al. ICLR 2025, this factor should be much smaller, but has no suggestion value
                wa_loss_II = self.reg_weight * 0.1 * torch.norm(orb_error_virt, p="fro") 
                reg_loss = wa_loss_I + wa_loss_II
            else:
                reg_loss = wa_loss_I 
        elif self.wa_type == 2:
            # modified from WALoss, add orbital gradient loss 
            orb_space_occ = torch.from_numpy(orb_coeffs[:, :nocc]).to(self._device).to(self._default_dtype)
            orb_space_virt = torch.from_numpy(orb_coeffs[:, nocc:]).to(self._device).to(self._default_dtype)
            ## waloss term for eigen space
            wa_error_occ = orb_space_occ.T @ (pred_fock - real_fock) @ orb_space_occ
            wa_error_virt = orb_space_virt.T @ (pred_fock - real_fock) @ orb_space_virt
            wa_loss = torch.norm(wa_error_occ, p="fro") + torch.norm(wa_error_virt, p="fro")
            ## orbital gradient term for off-diagonal space, analytically, this should be scaled by 2.0, but it is not necessary as we have a weight factor
            orb_grad = orb_space_virt.T @ (pred_fock - real_fock) @ orb_space_occ
            orb_grad_loss = self.reg_weight * torch.norm(orb_grad, p="fro") 
            reg_loss = wa_loss + orb_grad_loss
        else:
            raise ValueError(f"Unsupported wavefunction alignment type: {self.wa_type}")
        return reg_loss
    
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch) 
        
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device) 
            # forward propagation
            res = self.model(data)
            pred_pad_node, pred_pad_edge = res[0], res[1]
            real_pad_node, real_pad_edge = data.node_label, data.edge_label
            batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
            if hasattr(data, 'node_base'):
                pred_pad_node = pred_pad_node + data.node_base
            if hasattr(data, 'edge_base'):
                pred_pad_edge = pred_pad_edge + data.edge_base
            pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
            real_node, real_edge = real_pad_node[batch_mask_node], real_pad_edge[batch_mask_edge]
            batch_pred = torch.cat([pred_node, pred_edge], dim=0)
            batch_real = torch.cat([real_node, real_edge], dim=0)
            # calculate the loss 
            orig_loss = self.lossfn(batch_pred, batch_real)
            ## regularization term
            real_fock = self.mat_builder(real_pad_node, real_pad_edge, data.at_no, data.fc_edge_index)
            pred_fock = self.mat_builder(pred_pad_node, pred_pad_edge, data.at_no, data.fc_edge_index)
            reg_loss = self._get_reg_term(data, real_fock, pred_fock)
            ## total loss
            loss = orig_loss + reg_loss 
            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            # update EMA model
            if self.ema_model is not None:
                self.ema_model.update_parameters(self.model)
            # update learning rate
            if self.config.lr_scheduler != "plateau":
                with self.warmup_scheduler.dampening():
                    self.lr_scheduler.step()
            # for matrice, record the Frobenious norm distance 
            with torch.no_grad():
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
            # logging
            if (self.epoch % self.config.log_epoch == 0 and
                (step % self.config.log_step == 0 or
                 step == len(self.train_loader))):
                mae = self.meter.reduce()
                self.log.f.info(
                    "Epoch: [{iepoch:>4}][{step:>4}/{nstep}]   lr: {lr:3e}   train Error: node: {mae_n:10.7f},  edge: {mae_e:10.7f},  total: {mae_tot:10.7f}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae_n=mae[0],
                        mae_e=mae[1],
                        mae_tot=mae[2],
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                real_pad_node = data.node_label - data.node_base if hasattr(data, 'node_base') else data.node_label
                real_pad_edge = data.edge_label - data.edge_base if hasattr(data, 'edge_base') else data.edge_label
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
        mae = self.meter.reduce()
        self.log.f.info(f"EMA Validation Error: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.model.module, total_mae)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt") 

    def ema_validate(self):
        if self.ema_model is None:
            return
        self.ema_model.eval()
        self.meter.reset()
        with torch.no_grad():
            for data in self.valid_loader:
                data = data.to(self.device)
                res = self.ema_model(data)
                pred_pad_node, pred_pad_edge = res[0], res[1]
                real_pad_node = data.node_label - data.node_base if hasattr(data, 'node_base') else data.node_label
                real_pad_edge = data.edge_label - data.edge_base if hasattr(data, 'edge_base') else data.edge_label
                batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
                node_diff = (pred_pad_node - real_pad_node) * batch_mask_node
                edge_diff = (pred_pad_edge - real_pad_edge) * batch_mask_edge
                node_l2norm = torch.norm(node_diff, p="fro", dim=[1, 2]).sum()
                edge_l2norm = torch.norm(edge_diff, p="fro", dim=[1, 2]).sum()
                total_l2norm = node_l2norm + edge_l2norm
                num_node, num_edge = real_pad_node.shape[0], real_pad_edge.shape[0]
                self.meter.update(node_l2norm.item(), edge_l2norm.item(), total_l2norm.item(), num_node, num_edge, num_node + num_edge)
        mae = self.meter.reduce()
        self.log.f.info(f"EMA Validation Error: node: {mae[0]:10.7f}, edge: {mae[1]:10.7f}, total: {mae[2]:10.7f}")
        total_mae = mae[2]
        if self.config.lr_scheduler == "plateau":
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step(total_mae)
            self.early_stop(total_mae, self.best_l2fs[0].loss)
        self.save_best_k(self.ema_model.module, total_mae)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt")


class WavefunctionTrainer(OrbitalTrainer):
    def __init__(
        self, 
        model: nn.parallel.DistributedDataParallel, 
        config: NetConfig,
        device: torch.device, 
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler], 
        log: ZeroLogger,
    ):
        super(WavefunctionTrainer, self).__init__(model, config, device, train_loader, valid_loader, dist_sampler, log)
    
    def _get_reg_term(self, data, real_fock, pred_fock):
        real_fock = real_fock.cpu().numpy().astype(np.float64)
        at_no = data.at_no.cpu().numpy()
        coords = data.pos.cpu().numpy().astype(np.float64)
        charge = data.charge.to(torch.long).item()
        orb_coeffs, nocc, nvirt = self.orbital_calculator(real_fock, at_no, coords, charge)
        real_fock = torch.from_numpy(real_fock).to(self._device).to(self._default_dtype) 
        ## DIIS loss 
        cur_den = orb_coeffs[:, :nocc] @ orb_coeffs[:, :nocc].T
        cur_den = torch.from_numpy(cur_den).to(self._device).to(self._default_dtype)
        diis_error = cur_den @ pred_fock - pred_fock @ cur_den
        diis_loss = torch.norm(diis_error, p="fro")
        if self.wa_type == 0:
            ## Wavefunction Alignment loss for the eigen space 
            orb_space = torch.from_numpy(orb_coeffs).to(self._device).to(self._default_dtype)
            wa_error = orb_space.T @ (pred_fock - real_fock) @ orb_space
            wa_loss = torch.norm(wa_error, p="fro")
        elif self.wa_type == 1:
            ## Wavefunction Alignment loss for the occupied space and virtual space separately
            orb_occ = torch.from_numpy(orb_coeffs[:, :nocc]).to(self._device).to(self._default_dtype)
            orb_virt = torch.from_numpy(orb_coeffs[:, nocc:]).to(self._device).to(self._default_dtype)
            wa_error_occ = orb_occ.T @ (pred_fock - real_fock) @ orb_occ
            wa_loss_occ = torch.norm(wa_error_occ, p="fro")
            wa_error_virt = orb_virt.T @ (pred_fock - real_fock) @ orb_virt
            wa_loss_virt = torch.norm(wa_error_virt, p="fro")
            wa_loss = wa_loss_occ + wa_loss_virt
        elif self.wa_type == 2:
            # WALoss for full occupied space only 
            orb_occ = torch.from_numpy(orb_coeffs[:, :nocc]).to(self._device).to(self._default_dtype)
            wa_error_occ = orb_occ.T @ (pred_fock - real_fock) @ orb_occ
            wa_loss = torch.norm(wa_error_occ, p="fro")
        else:
            raise ValueError(f"Unsupported wavefunction alignment type: {self.wa_type}")
        # total loss
        reg_loss = self.reg_weight * diis_loss + wa_loss
        return reg_loss

