import torch
import torch.nn as nn
import einops
import numpy as np

from typing import List, Type
import math

from adapt3r.utils.point_cloud_utils import show_point_cloud_plt


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class MaxExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a max pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 **kwargs
                 ):
        """
        Args:
            in_shape: (int, int) or (int, int, int)
                - (n_points, in_channels) if 2D
                - (n_points, frame_stack, in_channels) if 3D
            out_channels: int
            use_layernorm: bool
            final_norm: str
        """
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        x = self.mlp(pc)
        if mask is not None:
            x.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        x = torch.max(x, 2)[0]

        x = self.final_projection(x)
        return x
    

class AttentionExtractor(nn.Module):
    """
    Extracts from a point cloud by passing through an MLP followed by a attention pooling.
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 hidden_dim=256,
                 num_heads=4,
                 **kwargs
                 ):
        super().__init__()
        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        # Create learnable key for each head
        self.queries = nn.Parameter(torch.randn(num_heads, hidden_dim))
        
        # Query projection MLP
        self.K_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Value projection MLP
        self.V_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(hidden_dim * num_heads, out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        
    def forward(self, pc, mask=None):
        # assume pc has dim batch, frame_stack, n, d
        B, F, N, D = pc.shape
        
        # Project to query and value
        K = self.K_mlp(pc)  # [B, F, N, hidden_dim]
        V = self.V_mlp(pc)  # [B, F, N, hidden_dim]
        
        # Expand key for broadcasting
        Q = self.queries.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
        
        # Compute attention using scaled_dot_product_attention
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(2)
        
        # Use scaled_dot_product_attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, F, N, num_heads, head_dim]
        
        # Combine heads
        x = einops.rearrange(attn_output, 'b f h d -> b f (h d)')  # [B, F, hidden_dim]
        
        # Final projection
        x = self.final_projection(x)  # [B, F, out_channels]
        return x
