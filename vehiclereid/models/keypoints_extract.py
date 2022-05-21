import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from .resnet import ResNet, BasicBlock
from .convnet_plus_classifier import ConvnetPlusClassifier


def marginal_logsoftmax(heatmap, dim):
    marginal = torch.mean(heatmap, dim=dim)
    sm = F.log_softmax(marginal, dim=2)
    return sm


def logprob_to_keypoints(prob, length):
    ruler = torch.log(torch.linspace(0, 1, length, device=prob.device)).type_as(prob).expand(1, 1, -1)
    return torch.sum(torch.exp(prob + ruler), dim=2, keepdim=True).squeeze(2)


def spacial_logsoftmax(heatmap, probs=False):
    height, width = heatmap.size(2), heatmap.size(3)
    hp, wp = marginal_logsoftmax(heatmap, dim=3), marginal_logsoftmax(heatmap, dim=2)
    hk, wk = logprob_to_keypoints(hp, height), logprob_to_keypoints(wp, width)
    if probs:
        return torch.stack((hk, wk), dim=2), (torch.exp(hp), torch.exp(wp))
    else:
        return torch.stack((hk, wk), dim=2)


def squared_diff(h, height):
    hs = torch.linspace(0, 1, height, device=h.device).type_as(h).expand(h.shape[0], h.shape[1], height)
    hm = h.expand(height, -1, -1).permute(1, 2, 0)
    hm = ((hs - hm) ** 2)
    return hm


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp[:, :, 0], height)
    wm = squared_diff(kp[:, :, 1], width)
    hm = hm.expand(width, -1, -1, -1).permute(1, 2, 3, 0)
    wm = wm.expand(height, -1, -1, -1).permute(1, 2, 0, 3)
    gm = - (hm + wm + eps).sqrt_() / (2 * sigma ** 2)
    gm = torch.exp(gm)
    return gm


class SpatialLogSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, heatmap, probs=True):
        return spacial_logsoftmax(heatmap, probs)


class GaussianLike(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, kp, height, width):
        return gaussian_like_function(kp, height, width, self.sigma)


class KeypointsExtractionModule_Resnet18(nn.Module):
    def __init__(self, soft_local_max_size=3, args=None):
        super(KeypointsExtractionModule_Resnet18, self).__init__()
        self.args = args
        base = ResNet(
            num_classes=10,
            loss={'xent'},
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            last_stride=1,
            strides=[2, 2, 2, 1],
            bn_neck=True,
            fc_dims=None,
            dropout_p=None)
        base.load_param('resnet18')
        self.model = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.maxpool, base.layer1, base.layer2,
            base.layer3, base.layer4
        )
        if self.args.use_rotation_prediction:
            self.aux_classifier = ConvnetPlusClassifier(args.rotation, BasicBlock, 512, 512)
        self.num_channels = 512
        self.soft_local_max_size = soft_local_max_size
        self.pad = self.soft_local_max_size // 2

    def forward(self, x, rot=False):
        batch_ = self.model(x)
        if rot:
            cls_score_rot = self.aux_classifier(batch_)
        else:
            cls_score_rot = None

        b = batch_.size(0)
        batch = F.relu(batch_)
        min_ = 1e-12
        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0].clamp(min=min_)
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        # local_max_score = exp / sum_exp.clamp(min=1e-5)
        local_max_score = exp / sum_exp.clamp(min=min_)
        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1).clamp(min=min_)
        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]
        # score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1).clamp(min=1e-5)
        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1).clamp(min=min_)
        # score = score * score.shape[2] * score[3]
        return score.unsqueeze(dim=1), cls_score_rot, batch_