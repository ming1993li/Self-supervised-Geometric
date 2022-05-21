import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .attention import get_attention_module_instance


def init_params(x):

    if x is None:
        return

    for m in x.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ABDModules(nn.Module):

    def __init__(self, owner, args, input_dim, atten_modules):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args.abd_dim
        self.args = args
        self.atten_modules = atten_modules
        self.use_dan = bool(atten_modules)
        self._init_reduction_layer()
        self._init_attention_modules()

    def _init_reduction_layer(self):

        reduction = nn.Sequential(
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        init_params(reduction)

        self.reduction = reduction

    def _init_attention_modules(self):

        args = self.args
        self.dan_module_names = set()
        DAN_module_names = self.atten_modules
        use_head = not args.abd_dan_no_head

        for name in DAN_module_names:
            if 'before_module' == name:
                before_module = get_attention_module_instance(
                    'identity',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(before_module)
                self.dan_module_names.add('before_module')
                self.before_module = before_module

            elif 'cam' == name:
                cam_module = get_attention_module_instance(
                    'cam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(cam_module)
                self.dan_module_names.add('cam_module')
                self.cam_module = cam_module

            elif 'pam' == name:
                pam_module = get_attention_module_instance(
                    'pam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(pam_module)
                self.dan_module_names.add('pam_module')
                self.pam_module = pam_module

            elif 'normed_cam' == name:
                normed_cam_module = get_attention_module_instance(
                    'normed_cam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(normed_cam_module)
                self.dan_module_names.add('normed_cam_module')
                self.normed_cam_module = normed_cam_module

            elif 'normed_pam' == name:
                normed_pam_module = get_attention_module_instance(
                    'normed_pam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(normed_pam_module)
                self.dan_module_names.add('normed_pam_module')
                self.normed_pam_module = normed_pam_module

            elif 'second_order_cam' == name:
                second_order_cam_module = get_attention_module_instance(
                    'second_order_cam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(second_order_cam_module)
                self.dan_module_names.add('second_order_cam_module')
                self.second_order_cam_module = second_order_cam_module

            elif 'second_order_pam' == name:
                second_order_pam_module = get_attention_module_instance(
                    'second_order_pam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(second_order_pam_module)
                self.dan_module_names.add('second_order_pam_module')
                self.second_order_pam_module = second_order_pam_module
            elif 'multi_order_cam' == name:
                multi_order_cam_module = get_attention_module_instance(
                    'multi_order_cam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(multi_order_cam_module)
                self.dan_module_names.add('multi_order_cam_module')
                self.multi_order_cam_module = multi_order_cam_module
            elif 'multi_order_pam' == name:
                multi_order_pam_module = get_attention_module_instance(
                    'multi_order_pam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(multi_order_pam_module)
                self.dan_module_names.add('multi_order_pam_module')
                self.multi_order_pam_module = multi_order_pam_module
            else:
                raise ValueError('The specified attention module name can not be found!')
        if not args.use_residual_in_abdmodul:
            sum_conv = nn.Sequential(
                nn.Dropout2d(0.1, False),
                nn.Conv2d(self.output_dim, self.input_dim, kernel_size=1),
            )
        else:
            sum_conv = nn.Sequential(
                nn.Dropout2d(0.1, False),
                nn.Conv2d(self.output_dim, self.input_dim, kernel_size=1),
                nn.BatchNorm2d(self.input_dim),
            )
            self.sum_relu = nn.ReLU(inplace=True)

        init_params(sum_conv)
        self.sum_conv = sum_conv

    def forward(self, x):

        out = self.reduction(x)
        if self.use_dan:
            to_sum = []
            # module_name: str
            for module_name in self.dan_module_names:
                x_out = getattr(self, module_name)(out)
                to_sum.append(x_out)
            if not self.args.use_residual_in_abdmodul:
                fmap_after = self.sum_conv(sum(to_sum))
            else:
                fmap_after = self.sum_conv(sum(to_sum))
                fmap_after = self.sum_relu(fmap_after + x)
        else:
            fmap_after = x

        return fmap_after

