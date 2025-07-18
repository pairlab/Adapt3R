import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel

from adapt3r.algos.base import ChunkPolicy
from adapt3r.algos.utils.diffusion_policy_utils.unet_modules import ConditionalUnet1D
from adapt3r.algos.utils.misc import weight_init


class DiffusionPolicyNetwork(nn.Module):
    """Wrapper module containing all networks for EMA management."""
    
    def __init__(self, diffusion_model, encoder, obs_proj):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.encoder = encoder
        self.obs_proj = obs_proj


class DiffusionPolicy(ChunkPolicy):
    def __init__(
        self,
        diffusion_model_factory,
        embed_dim,
        ema_factory,
        **kwargs
    ):
        super().__init__(**kwargs)

        diffusion_model = diffusion_model_factory(action_dim=self.network_action_dim)
        diffusion_model = diffusion_model.to(self.device)
        obs_channels = (
            self.encoder.n_out_perception * self.encoder.d_out_perception
            + self.encoder.n_out_lowdim * self.encoder.d_out_lowdim
        )
        obs_proj = nn.Linear(obs_channels, embed_dim)

        # Create wrapper module containing all networks
        self.networks = DiffusionPolicyNetwork(
            diffusion_model=diffusion_model,
            encoder=self.encoder,
            obs_proj=obs_proj
        ).to(self.device)

        # Initialize EMA for the entire networks module
        self.ema: EMAModel = ema_factory(parameters=self.networks.parameters())
        
        # Flag to track if we're using EMA parameters
        self._using_ema_params = False

        self.networks.diffusion_model.apply(weight_init)

    def train(self, mode=True):
        """Override train method to manage EMA parameters."""
        if mode and self._using_ema_params:
            # Switching to train mode, restore original parameters
            self.ema.restore(self.networks.parameters())
            self._using_ema_params = False
        elif not mode and not self._using_ema_params:
            # Switching to eval mode, use EMA parameters
            self.ema.store(self.networks.parameters())
            self.ema.copy_to(self.networks.parameters())
            self._using_ema_params = True
        
        # Call parent train method
        return super().train(mode)

    def eval(self):
        """Override eval method to manage EMA parameters."""
        return self.train(mode=False)

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)

        actions = data['actions']

        loss = self.networks.diffusion_model(cond, actions)
        
        # Update EMA after computing loss
        self.ema.step(self.networks.parameters())
        
        info = {
            "loss": loss.item(),
        }
        return loss, info

    def sample_actions(self, data):
        with torch.no_grad():
            data = self.preprocess_input(data, train_mode=False)
            cond = self.get_cond(data)
            actions = self.networks.diffusion_model.get_action(cond)
            return actions.cpu().numpy()

    def get_cond(self, data):
        # Use networks.encoder and networks.obs_proj
        perception_encodings, lowdim_encodings = self.obs_encode(data)
        perception_encodings = torch.cat(perception_encodings, dim=-1)
        lowdim_encodings = torch.cat(lowdim_encodings, dim=-1)
        encodings_cat = torch.cat((perception_encodings, lowdim_encodings), dim=-1)
        obs_emb = self.networks.obs_proj(encodings_cat)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond

    @property
    def encoder(self):
        """Property to maintain compatibility with base class."""
        if hasattr(self, 'networks'):
            return self.networks.encoder
        return super().encoder
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({'ema':self.ema.state_dict()} if self.ema is not None else None)
        return state_dict
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        ema_state_dict = state_dict.pop('ema', None)    
        super().load_state_dict(state_dict, *args, **kwargs)
        if ema_state_dict is not None:
            self.ema.load_state_dict(ema_state_dict)

class DiffusionModel(nn.Module):
    def __init__(
        self,
        noise_scheduler,
        action_dim,
        global_cond_dim,
        diffusion_step_emb_dim,
        down_dims,
        chunk_size,
        diffusion_inf_steps,
        device,
    ):
        super().__init__()
        self.device = device
        self.net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_emb_dim,
            down_dims=down_dims,
        ).to(self.device)
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
        noisy_action = torch.randn(
            (cond.shape[0], self.chunk_size, self.action_dim), device=self.device
        )
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_inf_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.net(sample=naction, timestep=k, global_cond=cond)
            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        return naction
