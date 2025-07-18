import numpy as np
import torch


def pad_update_intrinsics(intrinsics: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Update batched camera intrinsics after symmetric padding on all sides.

    Args:
        intrinsics (torch.Tensor): (B, 3, 3) tensor of intrinsics.
        pad (int): Number of pixels padded equally on all sides.

    Returns:
        torch.Tensor: (B, 3, 3) tensor of updated intrinsics.
    """
    new_intrinsics = intrinsics.clone()
    new_intrinsics[..., 0, 2] += pad  # cx
    new_intrinsics[..., 1, 2] += pad  # cy
    return new_intrinsics


def crop_update_intrinsics(intrinsics: torch.Tensor, x_low: torch.Tensor, y_low: torch.Tensor) -> torch.Tensor:
    """
    Update batched camera intrinsics after cropping.

    Args:
        intrinsics (torch.Tensor): (B, 3, 3) tensor of intrinsics.
        crop_info (dict): Dictionary with keys "x_low" and "y_low", each a (B,) tensor.

    Returns:
        torch.Tensor: (B, 3, 3) tensor of updated intrinsics.
    """
    new_intrinsics = intrinsics.clone()
    new_intrinsics[..., 0, 2] -= x_low
    new_intrinsics[..., 1, 2] -= y_low
    return new_intrinsics


def resize_update_intrinsics(intrinsics, orig_size, new_size):
    # orig_height, orig_width = orig_size
    # new_height, new_width = new_size
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    factor_y, factor_x = np.array(new_size) / np.array(orig_size)

    fx_new = fx * factor_x
    fy_new = fy * factor_y
    cx_new = cx * factor_x
    cy_new = cy * factor_y

    return np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]])
