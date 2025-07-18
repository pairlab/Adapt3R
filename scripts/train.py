import os
import time
import hydra
import wandb
import logging
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn as nn
import adapt3r.utils.utils as utils
from pyinstrument import Profiler
from adapt3r.utils.logger import Logger
from pathlib import Path

# Disable scientific notation for numpy and torch
import numpy as np
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


OmegaConf.register_new_resolver("eval", eval, replace=True)
os.environ["WANDB_INIT_TIMEOUT"] = "300"

@hydra.main(config_path="../config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training

    # create model
    model = instantiate(cfg.algo.policy,
                        shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    # logger.info("Computing normalization statistics")
    norm_stats = utils.compute_norm_stats(cfg, model)

    dataset = utils.make_dataset(cfg)

    model.preprocess_dataset(dataset, use_tqdm=train_cfg.use_tqdm)
    train_dataloader = instantiate(
        cfg.train_dataloader, 
        dataset=dataset)
    
    # start training
    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers,
                                      total_steps=train_cfg.n_epochs*len(train_dataloader),
                                      **cfg.algo.scheduler_kwargs)

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logging
    logger = utils.setup_logger('training.log', experiment_dir)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {total_params:,}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    config_file = Path(experiment_dir) / 'config.yaml'
    OmegaConf.save(cfg, config_file)
    logger.info(f"Saved configuration to: {config_file}")

    start_epoch, steps, wandb_id = 0, 0, None
    if train_cfg.resume and len([x for x in os.listdir(experiment_dir) if 'pth' in x]) > 0:
        checkpoint_path = experiment_dir
    else: 
        checkpoint_path = cfg.checkpoint_path
    
    if checkpoint_path is not None:
        try:
            state_dict = utils.load_checkpoint(checkpoint_path, logger=logger)
            utils.soft_load_state_dict(model, state_dict['model'])
            
            logger.info('Loading optimizer and scheduler states')
            for optimizer, opt_state_dict in zip(optimizers, state_dict['optimizers']):
                optimizer.load_state_dict(opt_state_dict)
            for scheduler, sch_state_dict in zip(schedulers, state_dict['schedulers']):
                scheduler.load_state_dict(sch_state_dict)
            scaler.load_state_dict(state_dict['scaler'])
            start_epoch = state_dict['epoch']
            steps = state_dict['steps']
            wandb_id = state_dict['wandb_id']
            norm_stats = state_dict['norm_stats']
        except Exception as e:
            logger.warning(f'Failed to load checkpoint: {str(e)}, starting from scratch')
    else:
        logger.info('Starting from scratch')
    
    if start_epoch >= train_cfg.n_epochs:
        logger.info("Training already completed. Exiting.")
        exit(0)

    model.normalizer.fit(norm_stats)
    

    if cfg.rollout.enabled:
        env_runner = instantiate(cfg.task.env_runner)

    try:
        run = wandb.init(
            dir=experiment_dir,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=wandb_id,
            **cfg.logging
        )
        logger.info("Successfully initialized wandb")
    except Exception as e:
        logger.warning(f'wandb failed to initialize: {str(e)}. Running with logging disabled')
        log_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        log_cfg.update({'mode': 'disabled'})
        run = wandb.init(
            dir=experiment_dir,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=wandb_id,
            **log_cfg
        )

    do_first_eval = False
    metric_logger = Logger(train_cfg.log_interval)
    
    logger.info("Starting training loop")
    for epoch in range(start_epoch, train_cfg.n_epochs + 1):
        t0 = time.time()
        model.train()
        training_loss = 0.0
        if train_cfg.do_profile:
            profiler = Profiler()
            profiler.start()
            
        for idx, data in enumerate(tqdm(train_dataloader, disable=not train_cfg.use_tqdm)):
            data = utils.map_tensor_to_device(data, device)
            
            for optimizer in optimizers:
                optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=train_cfg.use_amp):
                loss, info = model.compute_loss(data)

            scaler.scale(loss).backward()
            
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            if train_cfg.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip
                )

            for optimizer in optimizers:
                scaler.step(optimizer)
            
            scaler.update()

            info.update({'epoch': epoch})
            if train_cfg.grad_clip is not None:
                info.update({
                    "grad_norm": grad_norm.item(),
                })
            lr = optimizers[0].param_groups[0]["lr"]
            info.update({"lr": lr})
            info = {cfg.logging_folder: info}
            training_loss += loss.item()
            steps += 1
            metric_logger.update(info, steps)

            [scheduler.step() for scheduler in schedulers]

            if train_cfg.cut and idx > train_cfg.cut:
                break

        if train_cfg.do_profile:
            profiler.stop()
            profiler.print()

        training_loss /= len(train_dataloader)
        t1 = time.time()
        epoch_time = (t1-t0)/60
        logger.info(
            f"Epoch {epoch:3d} | train loss: {training_loss:5.5f} | time: {epoch_time:4.2f} min"
        )

        model_checkpoint_name_latest = os.path.join(
            experiment_dir, f"multitask_model_latest.pth"
        )
        utils.save_state({
            'model': model,
            'optimizers': optimizers,
            'schedulers': schedulers,
            'norm_stats': norm_stats,
            'scaler': scaler,
            'epoch': epoch + 1,
            'steps': steps,
            'wandb_id': wandb.run.id,
            'experiment_dir': experiment_dir,
            'experiment_name': experiment_name,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, model_checkpoint_name_latest)
        
        if epoch % train_cfg.save_interval == 0:
            if cfg.training.save_all_checkpoints:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth"
                    )
            else:
                model_checkpoint_name_ep = os.path.join(
                        experiment_dir, f"multitask_model.pth"
                    )
            utils.save_state({
                'model': model,
                'optimizers': optimizers,
                'schedulers': schedulers,
                'norm_stats': norm_stats,
                'scaler': scaler,
                'epoch': epoch,
                'steps': steps,
                'wandb_id': wandb.run.id,
                'experiment_dir': experiment_dir,
                'experiment_name': experiment_name,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, model_checkpoint_name_ep)
            logger.info(f"Saved checkpoint at epoch {epoch}")

        if cfg.rollout.enabled and \
            (epoch > start_epoch or do_first_eval) and \
                epoch % cfg.rollout.interval == 0:
            model.eval()
            logger.info("Starting evaluation rollout")
            rollout_results = env_runner.run(model, 
                                             n_video=cfg.rollout.n_video, 
                                             do_tqdm=train_cfg.use_tqdm,
                                             fault_tolerant=False)
            model.train()
            success_rate = rollout_results['rollout']['overall_success_rate']
            envs_solved = rollout_results['rollout']['environments_solved']
            logger.info(
                f"Evaluation results - success rate: {success_rate:1.3f} | environments solved: {envs_solved}"
            )
            metric_logger.log(rollout_results, step=steps)

    logger.info("Training completed successfully")
    wandb.finish()

if __name__ == "__main__":
    main()
