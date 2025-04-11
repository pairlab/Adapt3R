import einops
import matplotlib.pyplot as plt
import adapt3r.utils.pytorch3d_transforms as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from adapt3r.algos.base import ChunkPolicy
from adapt3r.algos.utils.diffuser_actor_utils.encoder import Encoder
from adapt3r.algos.utils.diffuser_actor_utils.layers import (
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
    FFWRelativeSelfCrossAttentionModule,
    ParallelAttention,
)
from adapt3r.algos.utils.diffuser_actor_utils.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb,
)
from adapt3r.algos.utils.diffuser_actor_utils.utils import normalise_quat
import adapt3r.env.libero.utils as lu

from adapt3r.utils.point_cloud_utils import show_point_cloud


class DiffuserActor(ChunkPolicy):
    def __init__(
        self,
        backbone="clip",
        image_size=(256, 256),
        embedding_dim=60,
        num_vis_ins_attn_layers=2,
        use_instruction=True,
        fps_subsampling_factor=5,
        diffusion_timesteps=100,
        inference_timesteps=10,
        nhist=3,
        relative=False,
        lang_enhanced=False,
        do_crop=True,
        task_suite_name=None,
        task_benchmark_name=None,
        beefy=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._relative = relative
        self.use_instruction = use_instruction
        self.nhist = nhist
        self.inference_timesteps = inference_timesteps
        self.do_crop = do_crop
        assert self.abs_action, 'diffuser actor is only compatible with abs actions'
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            beefy=beefy,
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            nhist=nhist,
            lang_enhanced=lang_enhanced,
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.n_steps = diffusion_timesteps

        # Move modules to the correct device
        self.encoder = self.encoder.to(self.device)
        self.prediction_head = self.prediction_head.to(self.device)
        # self.gripper_loc_bounds = self.gripper_loc_bounds.to(self.device)

        if task_suite_name == "libero":
            boundaries = lu.get_boundaries(benchmark_name=task_benchmark_name, tight=True)
        elif task_suite_name == 'metaworld':
            boundaries = torch.tensor(((-1, -1, -1), (1, 1, 1)))
            boundaries = einops.repeat(boundaries, "i j -> 50 i j")
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        # This just helps with pylance
        self.boundaries = self.boundaries

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        gt_trajectory, rgb_obs, pcd_obs, instruction, curr_gripper = self.parse_batch(data)
        loss = self.forward(
            gt_trajectory=gt_trajectory,
            rgb_obs=rgb_obs,
            pcd_obs=pcd_obs,
            instruction=instruction,
            curr_gripper=curr_gripper,
            run_inference=False,
            task_id=data["task_id"],
        )
        info = {"loss": loss.item()}
        return loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        _, rgb_obs, pcd_obs, instruction, curr_gripper = self.parse_batch(data)
        trajectory = self.forward(
            gt_trajectory=None,
            rgb_obs=rgb_obs,
            pcd_obs=pcd_obs,
            instruction=instruction,
            curr_gripper=curr_gripper,
            run_inference=True,
            task_id=data["task_id"],
        )
        actions = trajectory.detach()
        return actions

    def parse_batch(self, data):
        obs = data["obs"]

        if "abs_actions" in data:
            gt_trajectory = data["abs_actions"]

            pos, rot, gripper = torch.split(gt_trajectory, [3, 6, 1], dim=-1)
            gripper = (1 - gripper) / 2
            gt_trajectory = torch.cat((pos, rot, gripper), dim=-1)
        else:
            gt_trajectory = None

        rgb_obs = []
        pcd_obs = []
        for key in obs.keys():
            if "rgb" in key:
                rgb_obs.append(obs[key])
            elif "pointcloud_full" in key:
                pcd_obs.append(obs[key])

        rgb_obs = torch.stack(rgb_obs, dim=1)[:, :, 0]
        pcd_obs = torch.stack(pcd_obs, dim=1)[:, :, 0]

        pcd_obs = einops.rearrange(pcd_obs, "b ncam h w c -> b ncam c h w")

        # Extract instruction embeddings (B, 512)
        instruction = data["task_emb"]
        instruction = instruction.unsqueeze(1)  # (B, 1, 512)

        eef_pos = obs["robot0_eef_pos"]  # Position of the end-effector
        hand_mat = obs["hand_mat"][..., :3, :3]  # Rotation matrices
        eef_axis_angle = pt.matrix_to_axis_angle(hand_mat)
        eef_rot = self.rotation_transformer.forward(eef_axis_angle)
        # eef_quat = matrix_to_quaternion(hand_mat)  # Convert to quaternions
        # TODO: eef_gripper has dimension 2 when I think it should have dimension 1.
        # Nevertheless, it gets removed so I don't think that it matters
        curr_gripper = torch.cat((eef_pos, eef_rot), dim=-1)

        # Extract gripper state (B, T, 1)

        return gt_trajectory, rgb_obs, pcd_obs, instruction, curr_gripper

    def forward(
        self,
        gt_trajectory,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False,
        task_id=None,
    ):
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 9:]
            gt_trajectory = gt_trajectory[..., :9]
        curr_gripper = curr_gripper[..., :9]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                rgb_obs, pcd_obs, instruction, curr_gripper, task_id=task_id
            )
        # Normalize all positions
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3], task_id)
        pcd_obs = torch.permute(
            self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2]), task_id), [0, 1, 4, 2, 3]
        )
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3], task_id)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, task_id=task_id
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),),
            device=noise.device,
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3], timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9], timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # Condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = self.policy_forward_pass(noisy_trajectory, timesteps, fixed_inputs)

        # Compute loss
        trans = pred[..., :3]
        rot = pred[..., 3:9]
        pos_rot_loss = 30 * F.l1_loss(trans, noise[..., :3], reduction="mean") + 10 * F.l1_loss(
            rot, noise[..., 3:9], reduction="mean"
        )
        if gt_trajectory.shape[-1] > 7:
            openess = pred[..., 9:]
            openess_loss = F.binary_cross_entropy_with_logits(openess, gt_openess)
        loss = pos_rot_loss + openess_loss
        return loss

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (context_feats, context, instr_feats, adaln_gripper_feats, fps_feats, fps_pos) = (
            fixed_inputs
        )

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos,
        )

    def encode_inputs(self, visible_rgb, visible_pcd, instruction, curr_gripper, task_id=None):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(visible_rgb, visible_pcd)
        # Keep only low-res scale
        context_feats = einops.rearrange(rgb_feats_pyramid[0], "b ncam c h w -> b (ncam h w) c")
        context = pcd_pyramid[0]
        device = context.device

        # We've already normalized the point cloud such that in-boundary points are in [-1, 1]
        if self.do_crop:
            B, n_pts, d_feat = context_feats.shape

            above_lower = torch.all(context > -1, dim=-1)
            below_upper = torch.all(context < 1, dim=-1)
            mask = torch.logical_and(above_lower, below_upper)

            indices = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
            indices_repeat_3 = einops.repeat(indices, "b n -> b n k", k=3)
            indices_repeat_feat = einops.repeat(indices, "b n -> b n k", k=d_feat)
            context = torch.scatter(
                input=torch.zeros((B, n_pts + 1, 3), device=device, dtype=context.dtype),
                index=indices_repeat_3,
                src=context,
                dim=1,
            )[:, 1:]
            context_feats = torch.scatter(
                input=torch.zeros((B, n_pts + 1, d_feat), device=device, dtype=context_feats.dtype),
                index=indices_repeat_feat,
                src=context_feats,
                dim=1,
            )[:, 1:]
            # Update mask so that it accurately reflects which points are masked out of the pointcloud
            mask = torch.all(context == 0, dim=-1)

        if self._relative:
            context, curr_gripper = self.convert2rel(context, curr_gripper)

        # Encode instruction (B, seq_len, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats, feats_mask=mask
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1), self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats,
            context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats,
            fps_pos,  # sampled visual features
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device
        )
        # Noisy condition data
        noise_t = (
            torch.ones((len(condition_data),), device=condition_data.device)
            .long()
            .mul(self.position_noise_scheduler.timesteps[0])
        )
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(condition_mask, noisy_condition_data, noise)

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs,
            )
            out = out  # Keep only last layer's output
            pos = self.position_noise_scheduler.step(
                out[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                out[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        trajectory = torch.cat((trajectory, out[..., 9:]), -1)

        return trajectory

    def compute_trajectory(
        self,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        task_id=None,
    ):
        # Normalize all positions
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(
            self.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2]), task_id), [0, 1, 4, 2, 3]
        )
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3], task_id)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper, task_id=task_id
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros((B, self.chunk_size, D), device=rgb_obs.device)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(cond_data, cond_mask, fixed_inputs)

        # Unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3], task_id)
        # Convert gripper status to probability
        trajectory[..., -1] = 1 - 2 * trajectory[..., -1].sigmoid()

        return trajectory

    def normalize_pos(self, pos, task_id):
        boundaries = self.boundaries[task_id]
        B = boundaries.shape[0]
        n_singleton = len(pos.shape) - 2
        pos_min = boundaries[:, 0].reshape([B] + [1] * n_singleton + [3])
        pos_max = boundaries[:, 1].reshape([B] + [1] * n_singleton + [3])
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos, task_id):
        boundaries = self.boundaries[task_id]
        B = boundaries.shape[0]
        n_singleton = len(pos.shape) - 2
        pos_min = boundaries[:, 0].reshape([B] + [1] * n_singleton + [3])
        pos_max = boundaries[:, 1].reshape([B] + [1] * n_singleton + [3])
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper


# Include DiffusionHead class as well
class DiffusionHead(nn.Module):

    def __init__(
        self,
        embedding_dim=60,
        num_attn_heads=8,
        use_instruction=False,
        nhist=3,
        lang_enhanced=False,
    ):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        # We just assume use of 6D rotations
        rotation_dim = 6  # continuous 6D

        # Encoders
        self.traj_encoder = nn.Linear(9, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList(
            [
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim,
                    n_heads=num_attn_heads,
                    self_attention1=False,
                    self_attention2=False,
                    cross_attention1=True,
                    cross_attention2=False,
                    rotary_pe=False,
                    apply_ffn=False,
                )
            ]
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # Interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim,
                num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True,
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # Interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim),
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # Interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1)
        )

    def forward(
        self,
        trajectory,
        timestep,
        context_feats,
        context,
        instr_feats,
        adaln_gripper_feats,
        fps_feats,
        fps_pos,
    ):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats,
                seq1_key_padding_mask=None,
                seq2=instr_feats,
                seq2_key_padding_mask=None,
                seq1_pos=None,
                seq2_pos=None,
                seq1_sem_pos=traj_time_pos,
                seq2_sem_pos=None,
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, "b l c -> l b c")
        context_feats = einops.rearrange(context_feats, "b l c -> l b c")
        adaln_gripper_feats = einops.rearrange(adaln_gripper_feats, "b l c -> l b c")
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3],
            traj_feats,
            context[..., :3],
            context_feats,
            timestep,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            instr_feats,
        )
        return torch.cat((pos_pred, rot_pred, openess_pred), -1)

    def prediction_head(
        self,
        gripper_pcd,
        gripper_features,
        context_pcd,
        context_features,
        timesteps,
        curr_gripper_features,
        sampled_context_features,
        sampled_rel_context_pos,
        instr_feats,
    ):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(timesteps, curr_gripper_features)

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs,
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation = self.predict_rot(features, rel_pos, time_embs, num_gripper, instr_feats)

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return position, rotation, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(curr_gripper_features, "npts b c -> b npts c")
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper, instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper, instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
