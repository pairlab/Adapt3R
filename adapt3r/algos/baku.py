import einops

import einops
import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal

from adapt3r.algos.base import ChunkPolicy

from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.utils.gpt import GPT


class Baku(ChunkPolicy):
    def __init__(
            self, 
            hidden_dim, 
            std, 
            frame_stack, 
            embed_dim, 
            policy_head, 
            **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.std = std
        self.language_proj_type = "mlp"  # mlp or identity
        self.frame_stack = frame_stack

        num_feat_per_step = (
            self.encoder.n_out_perception + self.encoder.n_out_proprio
        )

        # actor
        self.trunk = GPTTrunk(
            repr_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_feat_per_step=num_feat_per_step,
        ).to(self.device)

        action_dim = self.network_action_dim * self.chunk_size
        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, action_dim, hidden_size=hidden_dim, num_layers=2
            )
        else:
            raise NotImplementedError()

        self.trunk.apply(weight_init)
        self._action_head.apply(weight_init)

    def forward(self, data):
        img_encodings, lowdim_encodings = self.obs_encode(data)
        encodings = img_encodings + lowdim_encodings
        encodings_stacked = torch.stack(encodings, dim=2)
        lang_emb = self.get_task_emb(data)
        features = self.trunk(encodings_stacked, lang_emb)
        pred_action = self._action_head(
            features,
            self.std,
        )
        return pred_action

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)


        actions = data[self.action_key]

        # TODO: currently it doesn't work with frame_stack > 1 because it assumes that
        # for each stacked frame we have a corresponding action sequence starting from
        # there. To fix this, we need to sample an action for each stacked frame and
        # arrange them accordingly. I believe this is similar to vqbet
        pred_action = self(data)

        B, F, D = actions.shape
        actions = einops.rearrange(actions, "b t d -> b (t d)")
        # TODO: change this when you fix the above issue with frame stacking
        actions = einops.repeat(actions, "b d -> b 1 d")
        loss = self._action_head.loss_fn(pred_action, actions)
        info = {"loss": loss.item()}

        return loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        with torch.no_grad():
            pred_action = self(data)
            actions = einops.rearrange(pred_action.mean, 'b fs (t d) -> b t fs d', t=self.chunk_size)
            actions = actions[:, :, -1] # take the actions corresponding to the last frame in the frame stack
            
        return actions.cpu().numpy()


class GPTTrunk(nn.Module):
    def __init__(
        self,
        repr_dim,
        hidden_dim,
        num_feat_per_step=1,
    ):
        super().__init__()

        self._repr_dim = repr_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(repr_dim))

        # GPT model
        self._policy = GPT(
            block_size=65,
            input_dim=repr_dim,
            output_dim=hidden_dim,
            n_layer=8,
            n_head=4,
            n_embd=hidden_dim,
            dropout=0.1,
        )

    def forward(self, obs, prompt):
        B, F, T, D = obs.shape
        B, D = prompt.shape

        # insert action token at each self._num_feat_per_step interval
        action_token = einops.repeat(self._action_token, "d -> b f 1 d", b=B, f=F)
        obs = torch.cat([obs, action_token], dim=2)
        obs = einops.rearrange(obs, "b f t d -> b (f t) d")
        prompt = einops.rearrange(prompt, "b d -> b 1 d")
        obs = torch.cat([prompt, obs], dim=1)

        # get action features
        features = self._policy(obs)
        features = features[:, 1:]
        num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
        features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        return features


class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return loss


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
