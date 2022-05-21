from torch import nn

from .classifier import Classifier


class ConvnetPlusClassifier(nn.Module):
    def __init__(self, num_classes, block, inplanes, planes, stride=2, downsample=None):
        super().__init__()

        self.layers = nn.Sequential(
            block(inplanes, planes, stride=stride, downsample=nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))),
            block(planes * block.expansion, planes, stride=stride, downsample=nn.Sequential(
                nn.Conv2d(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))),
            Classifier(classifier_type='cosine', num_channels=planes * block.expansion, num_classes=num_classes, bias=False)
        )
        # self.layers = nn.Sequential(
        #     nn.Conv2d(inplanes, inplanes * 2, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.Conv2d(inplanes * 2, inplanes * 4, kernel_size=3, padding=1, stride=2, bias=False),
        #     Classifier(classifier_type='cosine', num_channels=inplanes * 4, num_classes=num_classes, bias=False)
        # )

    def forward(self, features):
        classification_scores = self.layers(features)
        return classification_scores

