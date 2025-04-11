import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from diffusers.training_utils import EMAModel

from adapt3r.algos.utils.diffusion_modules import ConditionalUnet1D
from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.base import ChunkPolicy
from adapt3r.algos.utils.diffusion_modules import ConditionalUnet1D
from adapt3r.algos.utils.misc import weight_init


class DiffusionPolicy(ChunkPolicy):
    def __init__(
        self,
        diffusion_model_factory,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        diffusion_model = diffusion_model_factory(action_dim=self.network_action_dim)
        self.diffusion_model = diffusion_model.to(self.device)
        obs_channels = (
            self.encoder.n_out_perception * self.encoder.d_out_perception
            + self.encoder.n_out_proprio * self.encoder.d_out_proprio
        )
        # TODO: this assumes frame_stack=1
        self.obs_proj = nn.Linear(obs_channels, embed_dim)

        self.diffusion_model.apply(weight_init)

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)

        actions = data[self.action_key]

        loss = self.diffusion_model(cond, actions)
        info = {
            "loss": loss.item(),
        }
        return loss, info

    def sample_actions(self, data):
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            cond = self.get_cond(data)
            actions = self.diffusion_model.get_action(cond)

        return actions.cpu().numpy()

    def get_cond(self, data):
        img_encodings, lowdim_encodings = self.obs_encode(data)
        encodings = img_encodings + lowdim_encodings
        encodings_cat = torch.cat(encodings, dim=-1)
        encodings_cat = einops.rearrange(encodings_cat, "b f d -> (b f) d")
        obs_emb = self.obs_proj(encodings_cat)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond


class DiffusionModel(nn.Module):
    def __init__(
        self,
        noise_scheduler,
        action_dim,
        global_cond_dim,
        diffusion_step_emb_dim,
        down_dims,
        ema_power,
        chunk_size,
        diffusion_inf_steps,
        device,
    ):
        super().__init__()
        self.device = device
        net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_emb_dim,
            down_dims=down_dims,
        ).to(self.device)
        self.ema = EMAModel(parameters=net.parameters(), decay=ema_power)
        self.net = net
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.diffusion_inf_steps = diffusion_inf_steps

    def forward(self, cond, actions):
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (cond.shape[0],), device=self.device
        ).long()
        noise = torch.randn(actions.shape, device=self.device)
        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        # predict the noise residual
        noise_pred = self.net(noisy_actions, timesteps, global_cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def get_action(self, cond):
        nets = self.net
        noisy_action = torch.randn(
            (cond.shape[0], self.chunk_size, self.action_dim), device=self.device
        )
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_inf_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets(sample=naction, timestep=k, global_cond=cond)
            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        return naction

    def ema_update(self):
        self.ema.step(self.net.parameters())
