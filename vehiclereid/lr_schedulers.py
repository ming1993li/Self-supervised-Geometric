from __future__ import absolute_import
from __future__ import print_function

from bisect import bisect_right
import math

import torch


def init_lr_scheduler(optimizer,
                      lr_scheduler='multi_step',  # learning rate scheduler
                      stepsize=[20, 40],  # step size to decay learning rate
                      gamma=0.1,  # learning rate decay
                      max_epoch=100,
                      step_epoch=10,
                      ):
    if lr_scheduler == 'single_step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize[0], gamma=gamma)

    elif lr_scheduler == 'multi_step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == 'warm_up':
        return WarmupMultiStepLR(optimizer, warmup_factor=0.01, warmup_iters=10, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == 'cosine_step':
        return CosineStepLR(optimizer, max_epochs=float(max_epoch), last_epoch=-1, step_epochs=step_epoch, gamma=1.0)

    elif lr_scheduler == 'warmup_cosine':
        return WarmupCosineLR(optimizer, max_epochs=float(max_epoch), warmup_epochs=10, last_epoch=-1)

    elif lr_scheduler == 'warmup_cosine_step':
        return WarmupCosineStepLR(optimizer, max_epochs=float(max_epoch), warmup_epochs=10, step_epochs=step_epoch, last_epoch=-1)

    elif lr_scheduler == 'warmup_cosine_cosine':
        return WarmupCosineCosineLR(optimizer, max_epochs=float(max_epoch), warmup_epochs=10, step_epochs=step_epoch, last_epoch=-1)

    elif lr_scheduler == 'cyclic_cosine':
        return CyclicCosineLR(optimizer, 25, last_epoch=-1)
    else:
        raise ValueError('Unsupported lr_scheduler: {}'.format(lr_scheduler))


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=0.01,
        warmup_iters=10,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                # warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                warmup_factor = self.warmup_factor + alpha * (1 - self.warmup_factor)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class CosineStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        step_epochs=2,
        gamma=0.3,
        eta_min=0,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.eta_min=eta_min
        self.step_epochs = step_epochs
        self.gamma = gamma
        self.last_cosine_lr = 0
        super(CosineStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.max_epochs - self.step_epochs:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch) / (self.max_epochs - self.step_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            lr = [self.gamma ** (self.step_epochs - self.max_epochs + self.last_epoch + 1) * base_lr for base_lr in self.last_cosine_lr]

        return lr


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        eta_min=1e-7,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min=eta_min
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
        return lr


class WarmupCosineStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        step_epochs=10,
        eta_min=1e-7,
        gamma=0.3,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.step_epochs = step_epochs
        self.last_cosine_lr = 0
        self.gamma = gamma
        super(WarmupCosineStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        elif self.last_epoch < self.max_epochs - self.step_epochs + 1:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.step_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            # lr = [self.gamma ** (self.step_epochs - self.max_epochs + self.last_epoch) * base_lr for base_lr in
            #       self.last_cosine_lr]
            lr = self.last_cosine_lr
        return lr


class WarmupCosineCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs=10,
        step_epochs=10,
        eta_min=1e-7,
        gamma=0.3,
        last_epoch=-1,
    ):
        self.max_epochs = max_epochs - 1
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.step_epochs = step_epochs
        self.last_cosine_lr = 0
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = [base_lr * (self.last_epoch+1) / (self.warmup_epochs + 1e-32) for base_lr in self.base_lrs]
        elif self.last_epoch < self.max_epochs - self.step_epochs + 1:
            lr = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.step_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
            self.last_cosine_lr = lr
        else:
            lr = [base_lr * (1 + math.cos(
                      math.pi * (self.last_epoch - self.max_epochs + self.step_epochs - 1) / (self.step_epochs - 1))) / 2
                  for base_lr in self.last_cosine_lr]
        return lr


class CyclicCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 cycle_epoch,
                 cycle_decay=0.7,
                 last_epoch=-1):
        self.cycle_decay = cycle_decay
        self.cycle_epoch = cycle_epoch
        self.cur_count = 0
        super(CyclicCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.cur_count = (self.last_epoch + 1) // self.cycle_epoch
        decay = self.cycle_decay ** self.cur_count
        return [base_lr * decay *
         (1 + math.cos(math.pi * (self.last_epoch % self.cycle_epoch) / self.cycle_epoch)) / 2
         for base_lr in self.base_lrs]