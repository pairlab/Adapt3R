# Adapted from https://github.com/nickgkan/3d_diffuser_actor/blob/master/diffuser_actor/utils/position_encodings.py

import math

import torch
import torch.nn as nn
import einops
import numpy as np


def posemb_sincos(pos: np.ndarray, embedding_dim: int, min_period: float, max_period: float) -> np.ndarray:
    """Computes sine-cosine positional embedding vectors for scalar positions using NumPy."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    
    fraction = np.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = np.outer(pos, 1.0 / period * 2 * np.pi)
    
    return np.concatenate([np.sin(sinusoid_input), np.cos(sinusoid_input)], axis=-1)


class SinusoidalPosEmb2(nn.Module):
    """
    Based on pi's emb from OpenPI Zero implementation.
    Computes sine-cosine positional embedding vectors for scalar positions.
    """

    def __init__(self, embedding_dim, min_period, max_period):
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

        self.embedding_dim = embedding_dim
        self.min_period = min_period
        self.max_period = max_period

        # Precompute fraction and store as a buffer
        fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
        self.register_buffer("period", min_period * (max_period / min_period) ** fraction)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        pos: Tensor of shape (B,)
        Returns: Tensor of shape (B, embedding_dim)
        """
        sinusoid_input = pos.unsqueeze(-1) * (2 * torch.pi / self.period)  # Shape (B, embedding_dim//2)
        return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class SinusoidalPosEmb3D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 6 == 0, 'dim must be divisible by 6'

    @torch.no_grad()
    def forward(self, x):
        device = x.device
        sixth_dim = self.dim // 6
        emb = math.log(10000) / (sixth_dim - 1)
        emb = torch.exp(torch.arange(sixth_dim, device=device) * -emb)
        emb = emb.view(1, 1, 1, -1)
        x = x.unsqueeze(-1)
        emb = x * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = einops.rearrange(emb, 'b n i j -> b n (i j)')
        return emb
    

class NeRFSinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 6 == 0, 'dim must be divisible by 6'

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        device = x.device
        n_steps = self.dim // 6
        max_freq = n_steps - 1

        freq_bands = torch.pow(torch.tensor(2, device=device), 
                               torch.linspace(0, max_freq, steps=n_steps, device=device))
        freq_bands = freq_bands.view(1, 1, 1, -1)
        x = x.unsqueeze(-1)
        emb = freq_bands * x
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = einops.rearrange(emb, '... i j -> ... (i j)')
        return emb


class RotaryPositionEncoding(nn.Module):
    def __init__(self, feature_dim, pe_type='Rotary1D'):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        bsize, npoint = x_position.shape
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=x_position.device)
            * (-math.log(10000.0) / (self.feature_dim)))
        div_term = div_term.view(1, 1, -1) # [1, 1, d]

        sinx = torch.sin(x_position * div_term)  # [B, N, d]
        cosx = torch.cos(x_position * div_term)

        sin_pos, cos_pos = map(
            lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
            [sinx, cosx]
        )
        position_code = torch.stack([cos_pos, sin_pos] , dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3D(RotaryPositionEncoding):

    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    @torch.no_grad()
    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device)
            * (-math.log(10000.0) / (self.feature_dim // 3))
        )
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(
            lambda feat: torch.stack([feat, feat], -1).view(bsize, npoint, -1),
            [sinx, cosx, siny, cosy, sinz, cosz]
        )

        position_code = torch.stack([
            torch.cat([cosx, cosy, cosz], dim=-1),  # cos_pos
            torch.cat([sinx, siny, sinz], dim=-1)  # sin_pos
        ], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class LearnedAbsolutePositionEncoding3D(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)


class LearnedAbsolutePositionEncoding3Dv2(nn.Module):
    def __init__(self, input_dim, embedding_dim, norm="none"):
        super().__init__()
        norm_tb = {
            "none": nn.Identity(),
            "bn": nn.BatchNorm1d(embedding_dim),
        }
        self.absolute_pe_layer = nn.Sequential(
            nn.Conv1d(input_dim, embedding_dim, kernel_size=1),
            norm_tb[norm],
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        )

    def forward(self, xyz):
        """
        Arguments:
            xyz: (B, N, 3) tensor of the (x, y, z) coordinates of the points

        Returns:
            absolute_pe: (B, N, embedding_dim) tensor of the absolute position encoding
        """
        return self.absolute_pe_layer(xyz.permute(0, 2, 1)).permute(0, 2, 1)
