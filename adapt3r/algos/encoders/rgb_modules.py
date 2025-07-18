"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import einops

import numpy as np


###############################################################################
#
# Modules related to encoding visual information (can conditioned on language)
#
###############################################################################

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.

    From ACT codebase
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PatchEncoder(nn.Module):
    """
    A patch encoder that does a linear projection of patches in a RGB image.
    """

    def __init__(
        self, input_shape, patch_size=[16, 16], embed_size=64, no_patch_embed_bias=False
    ):
        super().__init__()
        C, H, W = input_shape
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn(x)
        return x


class SpatialSoftmax(nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = F.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoints = torch.cat([keypoint_x, keypoint_y], dim=1)
        return keypoints


class SpatialProjection(nn.Module):
    def __init__(self, input_shape, out_dim):
        super().__init__()

        assert (
            len(input_shape) == 3
        ), "[error] spatial projection: input shape is not a 3-tuple"
        in_c, in_h, in_w = input_shape
        num_kp = out_dim // 2
        self.out_dim = out_dim
        self.spatial_softmax = SpatialSoftmax(in_c, in_h, in_w, num_kp=num_kp)
        self.projection = nn.Linear(num_kp * 2, out_dim)

    def forward(self, x):
        out = self.spatial_softmax(x)
        out = self.projection(out)
        return out

    def output_shape(self, input_shape):
        return input_shape[:-3] + (self.out_dim,)


class ResnetEncoder(nn.Module):
    """
    A Resnet-18-based encoder for mapping an image to a latent vector
    From https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/lifelong/models/modules/rgb_modules.py

    Encode (f) an image into a latent vector.

    y = f(x), where
        x: (B, C, H, W)
        y: (B, H_out)

    Args:
        input_shape:      (C, H, W), the shape of the image
        output_size:      H_out, the latent vector size
        pretrained:       whether use pretrained resnet
        freeze: whether   freeze the pretrained resnet
        remove_layer_num: remove the top # layers
        no_stride:        do not use striding
    """

    def __init__(
        self,
        input_shape,
        output_size=None,
        pretrained=False,
        freeze=False,
        remove_layer_num=2,
        no_stride=False,
        language_dim=768,
        language_fusion="film",
        do_projection=True,
        return_all_feats = False,
        replace_bn_with_gn=False,
    ):

        super().__init__()

        # TODO: support for pretrained weights
        self.return_all_feats = return_all_feats

        ### 1. encode input (images) using convolutional layers
        assert remove_layer_num <= 5, "[error] please only remove <=5 layers"
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        layers = list(torchvision.models.resnet18(weights=weights).children())
        if remove_layer_num > 0:
            layers = layers[:-remove_layer_num]
        self.remove_layer_num = remove_layer_num
        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        in_channels = input_shape[0]
        if in_channels != 3:  # has depth, increase channel size
            assert in_channels == 4, 'other than RGBD not supported'
            conv0 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            # orig_layer = layers[0]
            # with torch.no_grad():
            #     # Initialize RGB channels with pretrained weights
            #     conv0.weight[:, :3] = orig_layer.weight.clone()
            #     # conv0.weight[:, 3] = orig_layer.weight.mean(dim=1)
            layers[0] = conv0

        self.no_stride = no_stride
        if self.no_stride:
            layers[0].stride = (1, 1)
            layers[3].stride = 1

        self.resnet18_base = nn.Sequential(*layers[:-2])
        self.block_1 = layers[-2][0]
        self.block_2 = layers[-2][1]
        self.block_3 = layers[-1][0]
        self.block_4 = layers[-1][1]
        
        self.language_fusion = language_fusion
        if language_fusion != "none":
            proj_1_2_channels = self.block_2.conv1.in_channels
            proj_3_4_channels = self.block_4.conv1.in_channels

            self.lang_proj1 = nn.Linear(language_dim, proj_1_2_channels * 2)
            self.lang_proj2 = nn.Linear(language_dim, proj_1_2_channels * 2)
            self.lang_proj3 = nn.Linear(language_dim, proj_3_4_channels * 2)
            self.lang_proj4 = nn.Linear(language_dim, proj_3_4_channels * 2)

        if freeze:
            if in_channels != 3:
                raise Exception(
                    "[error] cannot freeze pretrained "
                    + "resnet with the extra eye_in_hand input"
                )
            for param in self.resnet18_embeddings.parameters():
                param.requires_grad = False

        if pretrained:
            mean = [0.485, 0.456, 0.406] + [0] * (in_channels - 3)
            std = [0.229, 0.224, 0.225] + [1] * (in_channels - 3)
            self.normalizer = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalizer = nn.Identity()
                                    

        x = torch.zeros(1, *input_shape)
        y = self.block_4(
            self.block_3(self.block_2(self.block_1(self.resnet18_base(x))))
        )
        output_shape = y.shape  # compute the out dim
        self.do_projection = do_projection
        if do_projection:
            ### 2. project the encoded input to a latent space
            self.projection_layer = SpatialProjection(output_shape[1:], output_size)
            self.out_channels = self.projection_layer(y).shape[1]
            self.n_out = 1
        else:
            if y.shape[-3] != output_size:
                self.projection_layer = nn.Linear(y.shape[-3], output_size)
            else:
                self.projection_layer = nn.Identity()
            self.out_channels = output_size
            self.n_out = np.prod(y.shape[-2:])

        if replace_bn_with_gn:
            # Replace BatchNorm layers with GroupNorm
            self.resnet18_base = batch_norm_to_group_norm(self.resnet18_base)
            self.block_1 = batch_norm_to_group_norm(self.block_1)
            self.block_2 = batch_norm_to_group_norm(self.block_2)
            self.block_3 = batch_norm_to_group_norm(self.block_3)
            self.block_4 = batch_norm_to_group_norm(self.block_4)

    def forward(self, x, langs=None):
        x = self.normalizer(x)
        x = self.resnet18_base(x)

        out = {} if self.return_all_feats else None
        if out is not None: out['res1'] = x
    
        x = self.block_1(x)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = x.shape
            beta, gamma = torch.split(
                self.lang_proj1(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
        if out is not None: out['res2'] = x

        x = self.block_2(x)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = x.shape
            beta, gamma = torch.split(
                self.lang_proj2(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
        if out is not None: out['res3'] = x

        x = self.block_3(x)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = x.shape
            beta, gamma = torch.split(
                self.lang_proj3(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
        if out is not None: out['res4'] = x

        x = self.block_4(x)
        if langs is not None and self.language_fusion != "none":  # FiLM layer
            B, C, H, W = x.shape
            beta, gamma = torch.split(
                self.lang_proj4(langs).reshape(B, C * 2, 1, 1), [C, C], 1
            )
            x = (1 + gamma) * x + beta
        if out is not None: out['res5'] = x

        if self.do_projection:
            x = self.projection_layer(x)
            x = einops.rearrange(x, 'b d -> b 1 d')
            if out is not None: out['final'] = x
        else:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')
            x = self.projection_layer(x)
            if out is not None: out['final'] = x

        return out if self.return_all_feats else x

    # def output_shape(self, input_shape, shape_meta):
    #     return self.output_shape
  

class DINOEncoder(nn.Module):

    def __init__(
        self,
        input_shape,
        output_size,
        pretrained=True,
    ):
        super().__init__()
        assert (
            len(input_shape) == 3
        ), "[error] input shape of resnet should be (C, H, W)"

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=3),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.mlp_block = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Dropout(0.1),
        )
        self.projection = nn.Linear(384, output_size)
        self.output_shape = output_size
    
        if pretrained:
            for param in self.dino.parameters():
                param.requires_grad = False

    def forward(self, x, langs=None):
        x = self.preprocess(x)
        x = self.dino(x,is_training=True)['x_norm_patchtokens']
        mask = self.mlp_block(x).permute(0, 2, 1)
        mask = F.softmax(mask, dim=-1)
        x = torch.einsum('...si,...id->...sd', mask, x)
        x = self.projection(x)
        return x

    def output_shape(self, input_shape, shape_meta):
        return self.output_shape


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """

    # num_channels: num_groups
    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(
                        GROUP_NORM_LOOKUP[num_channels], num_channels
                    )
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer