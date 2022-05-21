import numpy as np
import torch
import torch.nn.functional as F
import copy


def apply_2d_rotation(input_tensor, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.

    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor.dim() >= 2

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )


def randomly_rotate_images(images):
    """Randomly rotates each image in the batch by 0, 90, 180, or 270 degrees."""
    batch_size = images.size(0)
    labels_rot = torch.from_numpy(np.random.randint(0, 4, size=batch_size))
    labels_rot = labels_rot.to(images.device)

    for r in range(4):
        mask = labels_rot == r
        images_masked = images[mask].contiguous()
        images[mask] = apply_2d_rotation(images_masked, rotation=r * 90)

    return images, labels_rot


def randomly_rotate_images_3(images):
    """Randomly rotates each image in the batch by 90, 180, or 270 degrees."""
    batch_size = images.size(0)
    labels_rot = torch.from_numpy(np.random.randint(1, 4, size=batch_size))
    labels_rot = labels_rot.to(images.device)

    for r in range(1, 4):
        mask = labels_rot == r
        images_masked = images[mask].contiguous()
        images[mask] = apply_2d_rotation(images_masked, rotation=r * 90)

    return images

