import torch
import torch.nn as nn
import einops
import numpy as np

from adapt3r.utils.point_cloud_utils import show_point_cloud
import matplotlib.pyplot as plt


class PointNetEncoder(nn.Module):
    """
    Encoder for Pointcloud

    Stolen from DP3 codebase
    """

    def __init__(self,
                 in_shape,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 block_channel=(64, 128, 256, 512),
                 reduction='max',
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.reduction = reduction
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
        
        if reduction == 'learned':
            self.red_mlp = nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.LayerNorm(64) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.LayerNorm(16) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.LayerNorm(4) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
        
        elif reduction == 'attention':
            self.key = nn.Parameter(torch.randn(256))
            self.Q_mlp = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256) if use_layernorm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(256, 256)
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
         
    def forward(self, xyz, pc, mask=None, pc_pos=None, pc_rgb=None, vis_attn=False):
        # assume pc has dim batch, frame_stack, n, d
        x = self.mlp(pc)
        if self.reduction == 'max':
            if mask is not None:
                x = x - 100000 * ~mask.unsqueeze(-1)
            x = torch.max(x, 2)[0]
        elif self.reduction == 'learned':
            logits = self.red_mlp(pc)
            if mask is not None:
                logits = logits - 100000 * ~mask.unsqueeze(-1)
            weights = torch.softmax(logits, 2)
            weighted_x = x * weights
            x = torch.sum(weighted_x, dim=2)
        elif self.reduction == 'attention':
            Q = self.Q_mlp(pc)
            logits = torch.einsum('bfnd,d->bfn', Q, self.key) / 16
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(logits, dim=-1)
            x = torch.einsum('bfn,bfnd->bfd', weights, x)

            if vis_attn:
                pc = pc_pos[0]
                rgb = pc_rgb[0]
                # Choose a colormap
                cmap = plt.get_cmap('viridis')
                # Map the data to colors
                # colors = cmap(logits[0, 0].cpu().numpy())[..., :3]
                weights_norm = weights[0, 0].cpu().numpy()
                weights_norm = (weights - weights_norm.min()) / (weights_norm.max())
                colors = cmap(weights_norm[0, 0].cpu().numpy())[..., :3]
                show_point_cloud(pc, rgb)
                show_point_cloud(pc, colors)
                show_point_cloud(pc, rgb)
            
        x = self.final_projection(x)
        return x


class iDP3Encoder(nn.Module):
    def __init__(self, 
                 in_shape,
                 out_channels: int=1024,
                 reduction='max',
                 h_dim=256, num_layers=4,
                 **kwargs):
        super().__init__()

        if type(in_shape) == int:
            in_channels = in_shape
        else:
            in_channels = in_shape[-1]
        self.out_channels = out_channels

        self.V_model = PCConvBlock(in_channels=in_channels, 
                                   out_channels=h_dim,
                                   h_dim=h_dim,
                                   num_layers=num_layers)
        self.h_dim = h_dim

        self.reduction = reduction
        if reduction == 'attention':
            self.key = nn.Parameter(torch.randn(h_dim))
            self.Q_model = PCConvBlock(in_channels=in_channels, 
                                       out_channels=h_dim,
                                       h_dim=h_dim,
                                       num_layers=num_layers)
        self.final_projection = nn.Linear(h_dim, out_channels)

    def forward(self, xyz, pc, mask, **kwargs):
        x = self.V_model(pc)

        if self.reduction == 'max':
            if mask is not None:
                # mask = einops.repeat(mask, 'b n -> b 1 ')
                x = x - 100000 * ~mask.unsqueeze(-1)
            x = torch.max(x, 2)[0]
        elif self.reduction == 'attention':
            Q = self.Q_model(pc)
            logits = torch.einsum('bfnd,d->bfn', Q, self.key) / np.sqrt(self.h_dim)
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(logits, dim=-1)
            x = torch.einsum('bfn,bfnd->bfd', weights, x)
            
        x = self.final_projection(x)

        return x


class PCConvBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels: int=1024,
                 h_dim=128, 
                 num_layers=4,
                 **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(in_channels, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)


    def forward(self, x):
        # assume x has dim batch, frame_stack, n, d
        B, F, N, D = x.shape
        x = einops.rearrange(x, 'b f n d -> (b f) d n') # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)
        x = einops.rearrange(x, '(b f) d n -> b f n d', b=B)
        return x





