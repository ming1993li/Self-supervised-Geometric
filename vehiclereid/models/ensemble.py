import torch
import torch.nn as nn
import torch.nn.functional as F

from .keypoints_extract import KeypointsExtractionModule_Resnet18, UnsupervisedLankmarksModule_Resnet18
from .branches import ABDModules
from .resnet import ResNet, Bottleneck, BasicBlock
from .classifier import CircleClassifier
from .gem import GeM

from copy import deepcopy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ResNet50_Kp_Atten(nn.Module):
    def __init__(self, num_classes=576, loss={'xent'}, pretrained=True,
                 soft_local_max_size=7, use_bnneck=True, **kwargs):
        super(ResNet50_Kp_Atten, self).__init__()
        base = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            bn_neck=True,
            fc_dims=None,
            dropout_p=None,
            **kwargs)
        base.load_param('resnet50')
        self.args = kwargs['args']
        self.common_out_dim = 2048
        self.kp_branch = KeypointsExtractionModule_Resnet18(soft_local_max_size, kwargs['args'])

        self.shallow_branch = nn.Sequential(base.conv1, base.bn1, base.relu,
                                         base.maxpool, base.layer1, base.layer2)

        self.global_branch = nn.Sequential(base.layer3, base.layer4,)

        self.attention_branch = nn.Sequential(deepcopy(base.layer3), deepcopy(base.layer4), )

        if not self.args.gem_pooling:
            self.gap_kp = nn.AdaptiveAvgPool2d(1)
            self.gap_global = nn.AdaptiveAvgPool2d(1)
        else:
            self.gap_kp = GeM()
            self.gap_global = GeM()
        self.use_bnneck = use_bnneck
        if not self.use_bnneck:
            if not self.args.circle_classifier:
                self.classifier_kp = nn.Linear(self.common_out_dim, num_classes)
                self.classifier_global = nn.Linear(self.common_out_dim, num_classes)
            else:
                self.classifier_kp = CircleClassifier(self.common_out_dim, num_classes,
                                        s=self.args.cosine_scale, m=self.args.cosine_margin)
                self.classifier_global = CircleClassifier(self.common_out_dim, num_classes,
                                        s=self.args.cosine_scale, m=self.args.cosine_margin)
        elif self.use_bnneck:
            self.bottleneck_kp = nn.BatchNorm1d(self.common_out_dim)
            self.bottleneck_global = nn.BatchNorm1d(self.common_out_dim)
            self.bottleneck_kp.bias.requires_grad_(False)  # no shift
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_kp.apply(weights_init_kaiming)
            self.bottleneck_global.apply(weights_init_kaiming)
            if not self.args.circle_classifier:
                self.classifier_kp = nn.Linear(self.common_out_dim, num_classes, bias=False)
                self.classifier_global = nn.Linear(self.common_out_dim, num_classes, bias=False)
                self.classifier_kp.apply(weights_init_classifier)
                self.classifier_global.apply(weights_init_classifier)
            else:
                self.classifier_kp = CircleClassifier(self.common_out_dim, num_classes,
                                        s=self.args.cosine_scale, m=self.args.cosine_margin)
                self.classifier_global = CircleClassifier(self.common_out_dim, num_classes,
                                        s=self.args.cosine_scale, m=self.args.cosine_margin)

    def forward(self, x, label=None, rot=False):
        if rot:
            kp_map_rot, cls_score_rot, _ = self.kp_branch(x, rot=True)
            return cls_score_rot, kp_map_rot

        kp_map, _, of_features = self.kp_branch(x, rot=False)
        f_shallow = self.shallow_branch(x)

        f_g = self.global_branch(f_shallow)

        f_atten = self.attention_branch(f_shallow * kp_map)
        # f_atten = self.attention_branch(f_shallow)

        kp_feat = self.gap_kp(f_atten)  # (b, 2048, 1, 1)
        global_feat = self.gap_global(f_g)  # (b, 2048, 1, 1)
        kp_feat = kp_feat.view(kp_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if not self.use_bnneck:
            bn_feat_kp = kp_feat
            bn_feat_global = global_feat
        elif self.use_bnneck:
            bn_feat_kp = self.bottleneck_kp(kp_feat)  # normalize for angular softmax
            bn_feat_global = self.bottleneck_global(global_feat)  # normalize for angular softmax

        if self.training:
            if not self.args.circle_classifier:
                cls_score_kp = self.classifier_kp(bn_feat_kp)
                cls_score_global = self.classifier_global(bn_feat_global)
            else:
                cls_score_kp = self.classifier_kp(bn_feat_kp, label)
                cls_score_global = self.classifier_global(bn_feat_global, label)
            return (cls_score_kp, cls_score_global), (kp_feat, global_feat), kp_map, of_features  # global feature for triplet loss
            # return (cls_score_kp, cls_score_global), torch.cat((kp_feat, global_feat), dim=1), kp_map, of_features  # global feature for triplet loss
        else:
            return torch.cat((bn_feat_kp, bn_feat_global), dim=1), kp_map


def resnet50_kp_atten(num_classes, loss={'xent'}, pretrained=True, **kwargs):
    model = ResNet50_Kp_Atten(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
    return model