from typing import Mapping
import torch
import torch.nn as nn
import warnings

class Normalizer(nn.Module):
    def __init__(self, 
                 mode='limits',
                 output_min=-1, 
                 output_max=1,
                 ):
        super().__init__()

        self.mode = mode
        self.output_min = output_min
        self.output_max = output_max
        self.stats = None

    def fit(self, normalization_stats):
        self.stats = normalization_stats
        if normalization_stats is None:
            warnings.warn("Warning: passed in norm stats is None, will not normalize")
            self.mode = 'identity'

    def normalize(self, data, keys=('actions',)):
        data_out = dict(data)
        if self.mode == 'identity':
            return data
        elif self.mode == 'limits':
            for key in data:
                if key in self.stats and key in keys:
                    key_stats = self.stats[key]
                    ds_min = key_stats['min']
                    ds_max = key_stats['max']
                    if type(data[key]) == torch.Tensor:
                        ds_min = torch.tensor(ds_min, device=data[key].device)
                        ds_max = torch.tensor(ds_max, device=data[key].device)
                    # Handle case where min equals max element-wise
                    equal_mask = (ds_max == ds_min)
                    # For equal elements, map to middle of output range
                    mid_val = (self.output_max + self.output_min) / 2
                    # For non-equal elements, apply standard normalization
                    norm = torch.where(equal_mask,
                        mid_val,
                        (data[key] - ds_min) * (self.output_max - self.output_min) / (ds_max - ds_min) + self.output_min
                    )
                    data_out[key] = norm
        else:
            raise NotImplementedError()
        
        for key, value in data.items():
            if type(value) == dict:
                data_out[key] = self.normalize(value, keys=keys)

        return data_out

    def unnormalize(self, data):
        # TODO: update to handle variable keys
        data_out = dict(data)
        if self.mode == 'identity':
            return data
        elif self.mode == 'limits':
            for key in data:
                if key in self.stats:
                    key_stats = self.stats[key]
                    ds_min = key_stats['min']
                    ds_max = key_stats['max']
                    if type(data[key]) == torch.Tensor:
                        ds_min = torch.tensor(ds_min, device=data[key].device)
                        ds_max = torch.tensor(ds_max, device=data[key].device)
                    unnorm = (data[key] - self.output_min) \
                        * (ds_max - ds_min) \
                            / (self.output_max - self.output_min) \
                                + ds_min
                    data_out[key] = unnorm
        else:
            raise NotImplementedError()
        return data_out
