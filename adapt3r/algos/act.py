import torch
from adapt3r.algos.base import ChunkPolicy
from adapt3r.algos.utils.misc import weight_init
from adapt3r.algos.utils.act_utils.detr_vae import DETRVAE

class ACT(ChunkPolicy):
    def __init__(
            self, 
            act_model: DETRVAE,
            loss_fn,
            kl_weight,
            encoder_input=('perception', 'lowdim',),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.kl_weight = kl_weight
        
        self.act_model = act_model(
            action_dim=self.network_action_dim,
            n_perception_input=self.encoder.n_out_perception,
            n_lowdim_input=self.encoder.n_out_lowdim,
            encoder_input=encoder_input,
            )
        self.act_model = self.act_model.to(self.device)
        
        self.act_model.apply(weight_init)


    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        
        actions = data["actions"]

        perception_encodings, lowdim_encodings, lang_emb = self.get_embeddings(data)
        

        is_pad = torch.zeros((actions.shape[0], actions.shape[1]), device=self.device, dtype=torch.bool)
        pred_action, _, latent = self.act_model(lowdim_encodings, perception_encodings, lang_emb, actions, is_pad)

        l1_loss = self.loss_fn(pred_action, actions)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(latent[0], latent[1])
        loss = l1_loss + total_kld[0]*self.kl_weight
        info = {
            'l1_loss': l1_loss.item(),
            'total_kld': total_kld[0].item(),
            'mean_kld': mean_kld.item(),
            'total_loss': loss.item(),
        }
        return loss, info
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)

        perception_encodings, lowdim_encodings, lang_emb = self.get_embeddings(data)

        pred_action, _, _ = self.act_model(lowdim_encodings, perception_encodings, lang_emb)

        return pred_action.detach().cpu().numpy()


    def get_embeddings(self, data):
        perception_encodings, lowdim_encodings = self.obs_encode(data)
        perception_encodings = torch.stack(perception_encodings, dim=1)
        B = perception_encodings.shape[0]
        D = perception_encodings.shape[-1]
        if len(lowdim_encodings) == 0:
            lowdim_encodings = torch.zeros((B, 0, D), device=perception_encodings.device)
        else:
            lowdim_encodings = torch.stack(lowdim_encodings, dim=1)
        
        lang_emb = self.get_task_emb(data)

        return perception_encodings, lowdim_encodings, lang_emb






def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld