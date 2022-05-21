from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from ..utils.rotation_utils import apply_2d_rotation


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing, if False, epsilon = 0
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class EquivarianceConstraintLoss(nn.Module):
    def __init__(self, mode='l2', use_gpu=True):
        super(EquivarianceConstraintLoss, self).__init__()
        self.mode = mode
        self.use_gpu = use_gpu

    def forward(self, hp, hp_rot, label_rot):
        loss_l2 = 0.
        # loss_l1 = 0.
        loss_kl = 0.
        for r in range(4):
            mask = label_rot == r
            hp_masked = hp[mask].contiguous()
            hp_masked = apply_2d_rotation(hp_masked, rotation=r * 90)

            loss_l2 += torch.pow(hp_masked - hp_rot[mask], 2).sum()
            # loss_l1 += torch.abs(hp_masked - hp_rot[mask]).sum()
            loss_kl += (hp_masked * torch.log(hp_masked / hp_rot[mask].clamp(min=1e-9))).sum()

        loss_kl = loss_kl / hp.size(0)
        loss_l2 = loss_l2 / hp.nelement()
        # loss_l1 = loss_l1 / hp.nelement()
        return loss_kl * 0.4 + loss_l2 * 0.6
