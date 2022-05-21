from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss, EquivarianceConstraintLoss
from .hard_mine_triplet_loss import TripletLoss
from .of_penalty import OFPenalty


def DeepSupervision(criterion, xs, y, weights=[1.0, 1.0]):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    assert len(xs) == len(weights)
    for i in range(len(xs)):
        loss += criterion(xs[i], y) * weights[i]
    loss /= len(xs)
    return loss
