from __future__ import absolute_import

from .resnet import resnet50, resnet50_fc512, resnet50_bnneck
# from .ensemble import resnet50_keypoints, resnet50_kp_separate, resnet50_kp_concate
from .ensemble import resnet50_kp_atten, resnet50_18


__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_bnneck': resnet50_bnneck,
    'resnet50_fc512': resnet50_fc512,
    # 'resnet50_keypoints': resnet50_keypoints,
    # 'resnet50_kp_separate': resnet50_kp_separate,
    'resnet50_kp_atten': resnet50_kp_atten,
    'resnet50_18': resnet50_18
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *kargs, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*kargs, **kwargs)


