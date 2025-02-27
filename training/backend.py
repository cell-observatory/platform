import matplotlib

matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from timm.scheduler import create_scheduler_v2

from deepspeed import initialize
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb

import ray.train.torch as raytorch
from ray.train import Checkpoint, report, get_context, get_checkpoint

import logging
import ujson
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import nullcontext

from data import ao_dataset
from training import masking
from training.earlystopping import EarlyStoppingCallback

from models.encoder import Encoder
from models.convnext import ConvNeXtV2
from models.vit import ViT
from models.baseline import Baseline
from models.maskedautoencoder import MaskedAutoEncoder
from models.jepa import JEPA

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)

dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def is_main_process():
    return get_context().get_world_rank() == 0


def summarize_model(model: nn.Module, inputs: tuple, batch_size: int, logdir: Path):
    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=(1, *inputs[1:]),
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=0,
        mode='eval'
    )
    train_stats = summary(
        model=model,
        input_size=inputs,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=1,
        mode='train'
    )

    with Path(logdir / 'model.log').open('w') as f:
        f.write(str(model_stats))

    model_logbook['training_batch_size'] = batch_size
    model_logbook['input_bytes'] = model_stats.total_input
    model_logbook['total_params'] = model_stats.total_params
    model_logbook['trainable_params'] = model_stats.trainable_params
    model_logbook['param_bytes'] = model_stats.total_param_bytes

    model_logbook['eval_macs'] = model_stats.total_mult_adds
    model_logbook['training_macs'] = train_stats.total_mult_adds

    model_logbook['forward_pass_bytes'] = model_stats.total_output_bytes
    model_logbook['forward_backward_pass_bytes'] = train_stats.total_output_bytes

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + model_logbook['training_model_bytes']

    model_logbook['layers'] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                'macs': layer.macs,
                'params': max(layer.num_params, 0),
                'param_bytes': layer.param_bytes,
                'forward_pass_bytes': layer.output_bytes,
                'forward_backward_pass_bytes': layer.output_bytes * 2,  # x2 for gradients
                'output_shape': layer.output_size,
            }

    with Path(logdir / 'model_logbook.json').open('w') as f:
        ujson.dump(
            model_logbook,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def restore_model(config):
    try:  # check if model already exists
        checkpoints = [d for d in config['outdir'].rglob('checkpoint_*') if d.is_dir()]
        checkpoints.sort(key=os.path.getctime)
        logger.info(f"Available checkpoints: {checkpoints}")

        logger.info(f"{config['logdir'] / 'logbook.csv'}: {Path(config['logdir'] / 'logbook.csv').exists()}")
        training_history = pd.read_csv(config['logdir'] / 'logbook.csv', header=0, index_col=0).dropna(axis=0, how='any')
        logger.info(f"Training history\n{training_history}")

        latest_checkpoint = checkpoints[-1]
        starting_epoch = training_history.index.values[-1]

        overall_step = 0
        best_loss = training_history.loc[starting_epoch, 'loss']
        logger.info(f"Restoring from {latest_checkpoint} epoch {starting_epoch} with loss {best_loss}")

        starting_epoch += 1
        step_logbook = {}
        epoch_logbook = training_history.to_dict(orient='index')
        epoch_left = config['epochs'] - starting_epoch
        logger.info(epoch_logbook)

        logger.info(f"Epochs left {epoch_left}")
        restored = True

        if epoch_left == 0:
            return

    except Exception as e:
        restored = False
        latest_checkpoint = None
        logger.warning(e)
        logger.warning(f"No model found in {config['outdir']}")
        best_loss, overall_step, starting_epoch = np.inf, 0, 0
        step_logbook, epoch_logbook = {}, {}

    return restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook


def get_lr_scheduler(opt: torch.optim.Optimizer, steps_per_epoch: int, config: dict, decay: str = 'cosine'):
    if config['fixedlr']:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config['epochs'],
        )
        logger.info(f"Training steps: [{steps_per_epoch * config['epochs']}]")
    else:
        decay_epochs = config['epochs'] - (config['warmup'] + config['cooldown'])
        total_steps = config['epochs'] * steps_per_epoch
        warmup_steps = config['warmup'] * steps_per_epoch
        cooldown_steps = config['cooldown'] * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        logger.info(
            f"Training [{config['epochs']=}: {total_steps=}]"
            f"({config['warmup']=}: {warmup_steps=})"
            f"({config['cooldown']=}: {cooldown_steps=})"
            f"({decay_epochs=}: {decay_steps=})"
        )

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=opt,
            sched=decay,
            num_epochs=config['epochs'],
            warmup_epochs=config['warmup'],
            cooldown_epochs=config['cooldown'],
            decay_epochs=decay_epochs,
            min_lr=1e-8,
            warmup_lr=1e-8,
        )

    return scheduler


def get_optimizer(
    params,
    config: dict,
    optimizer: str,
    steps_per_epoch: int,
    deepspeed_scheduler: bool = False
):
    if optimizer == 'adamw':
        opt = FusedAdam(
            params,
            lr=config['lr'],
            weight_decay=config['wd'],
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    elif optimizer == 'lamb':
        opt = FusedLamb(
            params,
            lr=config['lr'],
            weight_decay=config['wd'],
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    if deepspeed_scheduler:
        decay_epochs = config['epochs'] - (config['warmup'] + config['cooldown'])
        total_steps = config['epochs'] * steps_per_epoch
        warmup_steps = config['warmup'] * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        logger.info(
            f"Training [{config['epochs']=}, {steps_per_epoch=}: {total_steps=}]"
            f"({config['warmup']=}: {warmup_steps=})"
            f"({decay_epochs=}: {decay_steps=})"
        )

        scheduler = WarmupCosineLR(
            optimizer=opt,
            total_num_steps=total_steps,
            warmup_num_steps=warmup_steps,
            warmup_min_ratio=0.0,
            cos_min_ratio=0.0001,
            warmup_type='linear',
        )

        return opt, scheduler
    else:
        return opt


def supervised(config: dict):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    train_dataloader = ao_dataset.collect_dataset(
        config['dataset'],
        metadata=False,
        modes=config['pmodes'],
        distribution=config['distribution'],
        embedding=config['embedding'],
        samplelimit=config['samplelimit'],
        max_amplitude=config['max_amplitude'],
        photons_range=(config['min_photons'], config['max_photons']),
        cpu_workers=config['cpu_workers'],
        gpu_workers=config['gpu_workers'],
        model_input_shape=config['inputs'],
        batch_size=config['batch_size'],
        dtype=dtypes[config['amp']],
    )
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / (config['gpu_workers'] * config['workers'])))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    if config['network'].startswith('convnext'):
        model = ConvNeXtV2(
            model_template=config['network'],
            input_shape=config['inputs'],
            modes=config['pmodes'],
            depths=config['repeats'],
            dims=config['heads'],
        )
        block = None
    elif config['network'].startswith('vit'):
        model = ViT(
            model_template=config['network'],
            input_shape=config['inputs'],
            embed_dim=config['hidden_size'],
            lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
            axial_patch_size=1,
            num_heads=config['heads'] if type(config['heads']) == int else config['heads'][0],
            depth=config['repeats'] if type(config['repeats']) == int else config['repeats'][0],
            modes=config['pmodes'],
            proj_drop_rate=config['dropout'],
            fixed_dropout_depth=config['fixed_dropout_depth'],
        )
        block = Encoder
    elif config['network'].startswith('baseline'):
        model = Baseline(
            model_template=config['network'],
            input_shape=config['inputs'],
            embed_dim=config['hidden_size'],
            lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
            axial_patch_size=1,
            num_heads=config['heads'] if type(config['heads']) == int else config['heads'][0],
            depth=config['repeats'] if type(config['repeats']) == int else config['repeats'][0],
            modes=config['pmodes'],
            proj_drop_rate=config['dropout'],
            fixed_dropout_depth=config['fixed_dropout_depth'],
        )
        block = Encoder
    else:
        raise Exception(f'Network "{config["network"]}" is unknown.')

    summarize_model(
        model=model,
        inputs=config['inputs'],
        batch_size=config['batch_size'],
        logdir=config['logdir'],
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config,
        optimizer=config['opt'],
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    if config['finetune'] is not None:
        logger.info(f"Finetuning pretrained model @ {config['finetune']}")
        model_state = torch.load(config['finetune'].glob("*model.bin"))
        model.load_state_dict(model_state)

        optimizer_state = torch.load(config['finetune'].glob("*optimizer.bin"))
        opt.load_state_dict(optimizer_state)

    elif restored:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpointdir:
                logger.info(f"Loading pretrained model @ {latest_checkpoint} -> {checkpointdir}")

                model_state = torch.load(config['checkpointdir'] / f"best_model.bin")
                model.load_state_dict(model_state)

                optimizer_state = torch.load(config['checkpointdir'] / f"best_optimizer.bin")
                opt.load_state_dict(optimizer_state)

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=config['deepspeed_config'],
    )

    ray_context = get_context()
    loss_fn = nn.MSELoss(reduction='sum')
    mse_fn = nn.MSELoss(reduction='mean')
    loss_nans = 0
    with torch.autograd.set_detect_anomaly(True, check_nan=False):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config['profile'] else nullcontext() as profiler:

            for epoch in range(starting_epoch, config['epochs']):

                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss, mse = 0., 0.

                step_times, step_utilization, step_vram = [], [], []
                for step, batch in enumerate(train_dataloader):
                    inputs, zernikes = batch

                    step_time = time.time()
                    lr = opt.param_groups[0]["lr"]

                    outputs = model(inputs)
                    step_loss = loss_fn(outputs, zernikes)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    loss += step_loss.detach().float()
                    step_mse = mse_fn(outputs, zernikes)
                    mse += step_mse.detach().float()

                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss.detach().float(),
                        "step_mse": step_mse.detach().float(),
                        "step_lr": lr,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)
                with Path(config['logdir'] / 'memory.log').open('w') as f:
                    f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                mse = mse.item() / steps_per_epoch
                step_timer = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config['epochs'] - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization = np.mean(step_utilization)
                cuda_memory_allocated = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch:                 \t {epoch+1}/{config['epochs']}")
                logger.info(f"│ epoch_loss:                     \t {loss:.4g}")
                logger.info(f"│ epoch_mse:                      \t {mse:.4g}")
                logger.info(f"│ epoch_lr:                       \t {lr:.4g}")
                logger.info(f"│ cuda_utilization:               \t {cuda_utilization:.0f}%")
                logger.info(f"│ cuda_memory_allocated:          \t {cuda_memory_allocated:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated:      \t {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer:                     \t {step_timer * 1000:.0f}ms")
                logger.info(f"│ epoch_timer:                    \t {epoch_timer:.0f}s")
                logger.info(f"│ ETA:                            \t {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "mse": mse,
                    "lr": lr,
                    "cuda_utilization": cuda_utilization,
                    "cuda_memory_allocated": cuda_memory_allocated,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer,
                    "epoch_timer": epoch_timer,
                    "eta": eta,
                }
                df = pd.DataFrame.from_dict(epoch_logbook, orient='index')
                df.to_csv(config['logdir'] / 'logbook.csv')

                df = pd.DataFrame.from_dict(step_logbook, orient='index')
                df.to_csv(config['logdir'] / 'steplogbook.csv')

                with config['outdir'] / 'checkpoints' as checkpointdir:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(model.state_dict(), config['checkpointdir'] / f"best_model.bin")
                        torch.save(opt.state_dict(), config['checkpointdir'] / f"best_optimizer.bin")

                    checkpoint = Checkpoint.from_directory(checkpointdir)
                    report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)

                if is_main_process():
                    logger.info(epoch_logbook[epoch])

                if config['profile']:
                    profiler.step()

        with config['outdir'] / 'checkpoints' as checkpointdir:
            torch.save(model.state_dict(), config['checkpointdir'] / f"last_model.bin")
            torch.save(opt.state_dict(), config['checkpointdir'] / f"last_optimizer.bin")

            checkpoint = Checkpoint.from_directory(checkpointdir)
            report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)


def pixel_reconstruction(config: dict):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    train_dataloader = ao_dataset.collect_dataset(
        config['dataset'],
        metadata=False,
        modes=config['pmodes'],
        distribution=config['distribution'],
        embedding=config['embedding'],
        samplelimit=config['samplelimit'],
        max_amplitude=config['max_amplitude'],
        photons_range=(config['min_photons'], config['max_photons']),
        cpu_workers=config['cpu_workers'],
        gpu_workers=config['gpu_workers'],
        model_input_shape=config['inputs'],
        batch_size=config['batch_size'],
        dtype=dtypes[config['amp']],
    )
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / (config['gpu_workers'] * config['workers'])))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    if config['network'].startswith('mae'):
        model = MaskedAutoEncoder(
            model_template=config['network'],
            input_shape=config['inputs'],
            embed_dim=config['hidden_size'],
            lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
            axial_patch_size=1,
            num_heads=config['heads'] if type(config['heads']) == int else config['heads'][0],
            depth=config['repeats'] if type(config['repeats']) == int else config['repeats'][0],
            proj_drop_rate=config['dropout'],
            fixed_dropout_depth=config['fixed_dropout_depth'],
        )
        block = Encoder
    else:
        raise Exception(f'Network "{config["network"]}" is unknown.')

    summarize_model(
        model=model,
        inputs=config['inputs'],
        batch_size=config['batch_size'],
        logdir=config['logdir'],
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config,
        optimizer=config['opt'],
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    if config['finetune'] is not None:
        logger.info(f"Finetuning pretrained model @ {config['finetune']}")
        model_state = torch.load(config['finetune'].glob("*model.bin"))
        model.load_state_dict(model_state)

        optimizer_state = torch.load(config['finetune'].glob("*optimizer.bin"))
        opt.load_state_dict(optimizer_state)

    elif restored:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpointdir:
                logger.info(f"Loading pretrained model @ {latest_checkpoint} -> {checkpointdir}")

                model_state = torch.load(config['checkpointdir'] / f"best_model.bin")
                model.load_state_dict(model_state)

                optimizer_state = torch.load(config['checkpointdir'] / f"best_optimizer.bin")
                opt.load_state_dict(optimizer_state)

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=config['deepspeed_config'],
    )

    ray_context = get_context()
    loss_nans = 0
    with torch.autograd.set_detect_anomaly(True, check_nan=False):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config['profile'] else nullcontext() as profiler:

            for epoch in range(starting_epoch, config['epochs']):

                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.

                step_times, step_utilization, step_vram = [], [], []
                for step, batch in enumerate(train_dataloader):

                    inputs, zernikes = batch

                    step_time = time.time()
                    lr = opt.param_groups[0]["lr"]

                    step_loss = model(inputs)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    loss += step_loss.detach().float()

                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss,
                        "step_lr": lr,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)
                with Path(config['logdir'] / 'memory.log').open('w') as f:
                    f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config['epochs'] - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization = np.mean(step_utilization)
                cuda_memory_allocated = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch:                 \t {epoch+1}/{config['epochs']}")
                logger.info(f"│ epoch_loss:                     \t {loss:.4g}")
                logger.info(f"│ epoch_lr:                       \t {lr:.4g}")
                logger.info(f"│ cuda_utilization:               \t {cuda_utilization:.0f}%")
                logger.info(f"│ cuda_memory_allocated:          \t {cuda_memory_allocated:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated:      \t {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer:                     \t {step_timer * 1000:.0f}ms")
                logger.info(f"│ epoch_timer:                    \t {epoch_timer:.0f}s")
                logger.info(f"│ ETA:                            \t {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "epoch_lr": lr,
                    "cuda_utilization": cuda_utilization,
                    "cuda_memory_allocated": cuda_memory_allocated,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer,
                    "epoch_timer": epoch_timer,
                }
                df = pd.DataFrame.from_dict(epoch_logbook, orient='index')
                df.to_csv(config['logdir'] / 'logbook.csv')

                df = pd.DataFrame.from_dict(step_logbook, orient='index')
                df.to_csv(config['logdir'] / 'steplogbook.csv')

                with config['outdir'] / 'checkpoints' as checkpointdir:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(model.state_dict(), config['checkpointdir'] / f"best_model.bin")
                        torch.save(opt.state_dict(), config['checkpointdir'] / f"best_optimizer.bin")

                    checkpoint = Checkpoint.from_directory(checkpointdir)
                    report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)

                if is_main_process():
                    logger.info(epoch_logbook[epoch])

                if config['profile']:
                    profiler.step()

            with config['outdir'] / 'checkpoints' as checkpointdir:
                torch.save(model.state_dict(), config['checkpointdir'] / f"last_model.bin")
                torch.save(opt.state_dict(), config['checkpointdir'] / f"last_optimizer.bin")

                checkpoint = Checkpoint.from_directory(checkpointdir)
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)


def joint_embedding_prediction(config: dict):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    collate_fn = masking.MaskCollator(
        input_shape=config['inputs'],
        lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
        axial_patch_size=1,
        lateral_range=(0.2, 0.8),
        axial_range=(1.0, 1.0),
        num_blocks=8,
        patchify_scheme='blocks',
    )

    train_dataloader = ao_dataset.collect_dataset(
        config['dataset'],
        metadata=False,
        modes=config['pmodes'],
        distribution=config['distribution'],
        embedding=config['embedding'],
        samplelimit=config['samplelimit'],
        max_amplitude=config['max_amplitude'],
        photons_range=(config['min_photons'], config['max_photons']),
        cpu_workers=config['cpu_workers'],
        gpu_workers=config['gpu_workers'],
        model_input_shape=config['inputs'],
        batch_size=config['batch_size'],
        dtype=dtypes[config['amp']],
        # collate_fn=collate_fn
    )
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / (config['gpu_workers'] * config['workers'])))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    if config['network'].startswith('jepa'):
        model = JEPA(
            model_template=config['network'],
            input_shape=config['inputs'],
            embed_dim=config['hidden_size'],
            lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
            axial_patch_size=1,
            num_heads=config['heads'] if type(config['heads']) == int else config['heads'][0],
            depth=config['repeats'] if type(config['repeats']) == int else config['repeats'][0],
            proj_drop_rate=config['dropout'],
            fixed_dropout_depth=config['fixed_dropout_depth'],
        )
        block = Encoder
    else:
        raise Exception(f'Network "{config["network"]}" is unknown.')

    summarize_model(
        model=model,
        inputs=config['inputs'],
        batch_size=config['batch_size'],
        logdir=config['logdir'],
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config,
        optimizer=config['opt'],
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    # linearly increase lr from ema[0] to ema[1]
    total_steps = config['epochs'] * steps_per_epoch
    ema_scheduler = (
        config['ema'][0] + i * (config['ema'][1]-config['ema'][0]) / total_steps
        for i in range(total_steps+1)
    )

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=config['deepspeed_config'],
    )

    if config['finetune'] is not None:
        logger.info(f"Finetuning pretrained model @ {config['finetune']}")
        model_state = torch.load(config['finetune'].glob("*model.bin"))
        model.load_state_dict(model_state)

        optimizer_state = torch.load(config['finetune'].glob("*optimizer.bin"))
        opt.load_state_dict(optimizer_state)

    elif restored:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpointdir:
                logger.info(f"Loading pretrained model @ {latest_checkpoint} -> {checkpointdir}")

                model_state = torch.load(config['checkpointdir'] / f"best_model.bin")
                model.load_state_dict(model_state)

                optimizer_state = torch.load(config['checkpointdir'] / f"best_optimizer.bin")
                opt.load_state_dict(optimizer_state)

    es = EarlyStoppingCallback(min_delta=1e-4, patience=10)
    ray_context = get_context()
    loss_nans = 0

    with torch.autograd.set_detect_anomaly(True, check_nan=False):

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config['profile'] else nullcontext() as profiler:

            for epoch in range(starting_epoch, config['epochs']):

                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.

                step_times, step_utilization, step_vram = [], [], []
                for step, batch in enumerate(train_dataloader):
                    inputs, zernikes = batch

                    step_time = time.time()
                    lr = opt.param_groups[0]["lr"]

                    step_loss = model(inputs)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    model.ema_update(beta=next(ema_scheduler))

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    loss += step_loss.detach().float()

                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss,
                        "step_lr": lr,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)
                with Path(config['logdir'] / 'memory.log').open('w') as f:
                    f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config['epochs'] - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization = np.mean(step_utilization)
                cuda_memory_allocated = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch:                 \t {epoch+1}/{config['epochs']}")
                logger.info(f"│ epoch_loss:                     \t {loss:.4g}")
                logger.info(f"│ epoch_lr:                       \t {lr:.4g}")
                logger.info(f"│ cuda_utilization:               \t {cuda_utilization:.0f}%")
                logger.info(f"│ cuda_memory_allocated:          \t {cuda_memory_allocated:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated:      \t {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer:                     \t {step_timer * 1000:.0f}ms")
                logger.info(f"│ epoch_timer:                    \t {epoch_timer:.0f}s")
                logger.info(f"│ ETA:                            \t {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "epoch_lr": lr,
                    "cuda_utilization": cuda_utilization,
                    "cuda_memory_allocated": cuda_memory_allocated,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer,
                    "epoch_timer": epoch_timer,
                }
                df = pd.DataFrame.from_dict(epoch_logbook, orient='index')
                df.to_csv(config['logdir'] / 'logbook.csv')

                df = pd.DataFrame.from_dict(step_logbook, orient='index')
                df.to_csv(config['logdir'] / 'steplogbook.csv')

                with config['outdir'] / 'checkpoints' as checkpointdir:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(model.state_dict(), config['checkpointdir'] / f"best_model.bin")
                        torch.save(opt.state_dict(), config['checkpointdir'] / f"best_optimizer.bin")

                    checkpoint = Checkpoint.from_directory(checkpointdir)
                    report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)

                if is_main_process():
                    logger.info(epoch_logbook[epoch])

                if config['profile']:
                    profiler.step()

            with config['outdir'] / 'checkpoints' as checkpointdir:
                torch.save(model.state_dict(), config['checkpointdir'] / f"last_model.bin")
                torch.save(opt.state_dict(), config['checkpointdir'] / f"last_optimizer.bin")

                checkpoint = Checkpoint.from_directory(checkpointdir)
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)
