from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from PIL import Image
import os.path as osp
import copy
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms.functional as trans_f


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class ImageDataset_Rotation(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, rotation=8):
        self.dataset = dataset
        self.transform = transform
        self.rotation = rotation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        img_origin = self.transform[1](self.transform[0](copy.deepcopy(img)))
        img_rot = self.transform[0](img)

        label_rot = np.random.randint(0, self.rotation)
        img_rot = trans_f.rotate(img_rot, label_rot * 360 / self.rotation)
        img_rot = self.transform[1](img_rot)

        return img_origin, pid, camid, img_path, img_rot, label_rot