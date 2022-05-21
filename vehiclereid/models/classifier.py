import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

from .utils import utils as cutils
from .utils import tools


class Classifier(nn.Module):
    def __init__(self, classifier_type='cosine', num_channels=512, num_classes=4, bias=False):
        super().__init__()

        self.classifier_type = classifier_type
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.global_pooling = True

        if self.classifier_type == "cosine":
            self.layers = cutils.CosineClassifier(
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                scale=10.0,
                learn_scale=True,
                bias=bias,
            )

        elif self.classifier_type == "linear":
            self.layers = nn.Linear(self.num_channels, self.num_classes, bias=bias)
            if bias:
                self.layers.bias.data.zero_()

            fout = self.layers.out_features
            self.layers.weight.data.normal_(0.0, np.sqrt(2.0 / fout))

        elif self.classifier_type == "mlp_linear":
            mlp_channels = [int(num_channels / 2), int(num_channels / 4)]
            num_mlp_channels = len(mlp_channels)
            mlp_channels = [self.num_channels,] + mlp_channels
            self.layers = nn.Sequential()

            pre_act_relu = False
            if pre_act_relu:
                self.layers.add_module("pre_act_relu", nn.ReLU(inplace=False))

            for i in range(num_mlp_channels):
                self.layers.add_module(
                    f"fc_{i}", nn.Linear(mlp_channels[i], mlp_channels[i + 1], bias=False),
                )
                self.layers.add_module(f"bn_{i}", nn.BatchNorm1d(mlp_channels[i + 1]))
                self.layers.add_module(f"relu_{i}", nn.ReLU(inplace=True))

            fc_prediction = nn.Linear(mlp_channels[-1], self.num_classes)
            fc_prediction.bias.data.zero_()
            self.layers.add_module("fc_prediction", fc_prediction)
        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(self.classifier_type)
            )

    def flatten(self):
        return (
            self.classifier_type == "linear"
            or self.classifier_type == "cosine"
            or self.classifier_type == "mlp_linear"
        )

    def forward(self, features):
        if self.global_pooling:
            features = tools.global_pooling(features, pool_type="avg")

        if features.dim() > 2 and self.flatten():
            features = features.view(features.size(0), -1)

        scores = self.layers(features)
        return scores


class CircleClassifier(nn.Module):
    def __init__(self, in_features, out_features, s=256, m=0.25):
        super().__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self._s = s
        self._m = m
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, bn_feat, targets):
        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        one_hot = torch.zeros(sim_mat.size(), device=targets.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        pred_class_logits = one_hot * s_p + (1.0 - one_hot) * s_n

        return pred_class_logits