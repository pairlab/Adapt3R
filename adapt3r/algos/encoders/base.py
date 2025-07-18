import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(
        self,
        frame_stack=1,
        shape_meta=None,
        lang_embed_dim=None,
        **kwargs
    ):
        super().__init__()
        self.frame_stack = frame_stack
        self.shape_meta = shape_meta
        self.lang_embed_dim = lang_embed_dim

        # These need to be populated by subclasses
        self.n_out_perception = 0
        self.d_out_perception = 0
        self.n_out_lowdim = 0
        self.d_out_lowdim = 0

        if lang_embed_dim is not None:
            if shape_meta.task.type == "onehot":
                self.task_encoder = nn.Embedding(
                    num_embeddings=shape_meta.task.n_tasks, embedding_dim=lang_embed_dim
                )
            else:
                self.task_encoder = nn.Linear(shape_meta.task.dim, lang_embed_dim)

    def get_task_emb(self, data):
        if "task_emb" in data:
            return self.task_encoder(data["task_emb"])
        else:
            return self.task_encoder(data["task_id"])
