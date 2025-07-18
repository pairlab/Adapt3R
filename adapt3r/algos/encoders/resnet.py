# Adapted from https://github.com/nickgkan/3d_diffuser_actor/blob/master/diffuser_actor/utils/resnet.py

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Type, Union, List, Any
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck, ResNet


# TODO probably delete all the film stuff

def load_resnet50(pretrained: bool = False, film=False, language_dim=None):
    backbone = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def load_resnet18(pretrained: bool = False, film=False, language_dim=None):
    backbone = _resnet('resnet18', Bottleneck, [2, 2, 2, 2], pretrained=pretrained, progress=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return backbone, normalize

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    film=False, 
    language_dim=None,
    **kwargs: Any
) -> ResNet:
    model = ResNetFeatures(block, layers, film, language_dim, **kwargs)
    if pretrained:
        if int(torch.__version__[0]) <= 1:
            from torch.hub import load_state_dict_from_url
            from torchvision.models.resnet import model_urls
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        else:
            raise NotImplementedError("Pretrained models not supported in PyTorch 2.0+")
    return model


class ResNetFeatures(ResNet):
    def __init__(self, block, layers, film=False, language_dim=None, **kwargs):
        super().__init__(block, layers, **kwargs)

        self.film = film
        if film:
            self.lang_proj1 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj2 = nn.Linear(language_dim, 64 * 2)
            self.lang_proj3 = nn.Linear(language_dim, 128 * 2)
            self.lang_proj4 = nn.Linear(language_dim, 128 * 2)

    def _forward_impl(self, x: torch.Tensor, langs=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        # Exiting early to save compute. 
        return {'out': x1}
