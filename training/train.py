import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import torch
from ray import init, cluster_resources, shutdown
from ray.train import ScalingConfig,  CheckpointConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer, TorchConfig

import sys
import logging
import os
import time
from pathlib import Path

from training import backend
from utils import cli

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "GRAPH"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)

def parse_args(args):
    train_parser = cli.argparser()

    train_parser.add_argument(
        "--network", default='baseline-base', type=str, help="codename for target network to train"
    )

    train_parser.add_argument(
        "--dataset", type=Path, help="path to dataset directory"
    )

    train_parser.add_argument(
        "--outdir", default="../pretrained_models", type=Path, help='path to save trained models'
    )

    train_parser.add_argument(
        "--batch_size", default=2048, type=int, help="number of images per batch"
    )

    train_parser.add_argument(
        "--hidden_size", default=768, type=int, help="hidden size of transformer block"
    )

    train_parser.add_argument(
        "--patches", default='32-16-8-8', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--heads", default='2-4-8-16', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--repeats", default='2-4-6-2', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--input_shape", default=64, type=int, help="PSF input shape"
    )

    train_parser.add_argument(
        "--modes", default=15, type=int, help="number of modes to describe aberration"
    )

    train_parser.add_argument(
        "--pmodes", default=None, type=int, help="number of modes to predict"
    )

    train_parser.add_argument(
        "--min_photons", default=1, type=int, help="minimum photons for training samples"
    )

    train_parser.add_argument(
        "--max_photons", default=10000000, type=int, help="maximum photons for training samples"
    )

    train_parser.add_argument(
        "--dist", default='/', type=str, help="distribution of the zernike amplitudes"
    )

    train_parser.add_argument(
        "--embedding", default='spatial_planes', type=str, help="embedding option to use for training"
    )

    train_parser.add_argument(
        "--samplelimit", default=None, type=int, help="max number of files to load from a dataset [per bin/class]"
    )

    train_parser.add_argument(
        "--max_amplitude", default=1., type=float, help="max amplitude for the zernike coefficients"
    )

    train_parser.add_argument(
        '--fixedlr', action='store_true',
        help='toggle to use a fixed learning rate'
    )

    train_parser.add_argument(
        "--lr", default=5e-4, type=float,
        help='initial learning rate; optimal config: 1e-3 for LAMB and 5e-4 for AdamW'
    )

    train_parser.add_argument(
        "--wd", default=5e-5, type=float, help='initial weight decay; optimal config: 1e-2 for LAMB and 5e-6 for AdamW'
    )

    train_parser.add_argument(
        "--ld", default=None, type=float, help='optional layer decay'
    )

    train_parser.add_argument(
        "--ema", default=(.998, 1.), type=tuple, help='exponential moving average scaler'
    )

    train_parser.add_argument(
        "--clip_grad", default=.5, type=float, help='optional value to clip gradients'
    )

    train_parser.add_argument(
        "--dropout", default=0.1, type=float, help='initial dropout rate for stochastic depth'
    )

    train_parser.add_argument(
        "--opt", default='lamb', type=str, help='optimizer to use for training'
    )

    train_parser.add_argument(
        "--warmup", default=25, type=int, help='number of epochs for the initial linear warmup'
    )

    train_parser.add_argument(
        "--cooldown", default=50, type=int, help='number of epochs for the final linear cooldown'
    )

    train_parser.add_argument(
        "--epochs", default=500, type=int, help="number of training epochs"
    )

    train_parser.add_argument(
        "--workers", default=1, type=int, help='number worker nodes'
    )

    train_parser.add_argument(
        "--cpu_workers", default=1, type=int, help='number of CPU cores per worker'
    )

    train_parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs per worker'
    )

    train_parser.add_argument(
        '--fixed_dropout_depth', action='store_true',
        help='toggle to linearly increase dropout rate for deeper layers'
    )

    train_parser.add_argument(
        "--amp", type=str, default='fp16', choices=["no", "fp16", "bf16", "fp8"],
        help='optional toggle for automatic mixed precision training'
             '(https://www.tensorflow.org/guide/mixed_precision)'
    )

    train_parser.add_argument(
        "--finetune", default=None, type=Path,
        help='evaluate on validation set'
    )

    train_parser.add_argument(
        '--profile', action='store_true',
        help='toggle to profile the training process'
    )

    train_parser.add_argument(
        '--jepa', action='store_true',
        help='toggle to train using JEPA'
    )


    train_parser.add_argument(
        '--mae', action='store_true',
        help='toggle to train using JEPA'
    )

    return train_parser.parse_known_args(args)[0]


def train_model(
    network: str = 'baseline-base',
    dataset: Path = None,
    outdir: Path = '../pretrained_models',
    input_shape : int = 64,
    modes: int = 15,
    pmodes: int = None,
    batch_size: int = 512,
    hidden_size: int = 768,
    patches: str = 32, # '32-16-8-8' for multistage ,
    heads: str = 16, # '2-4-8-16' for multistage,
    repeats: int = 4, # '2-4-6-2' for multistage,
    opt: str = 'lamb',
    lr: float = 5e-4,
    wd: float = 5e-5,
    ld: float = None,
    ema: tuple = (.998, 1.),
    warmup: int = 25,
    cooldown: int = 50,
    clip_grad: float = .5,
    epochs: int = 500,
    fixedlr: bool = False,
    dropout: float = 0.1,
    fixed_dropout_depth: bool = False,
    amp: str = 'fp16',
    finetune: Path = None,
    jepa: bool = False,
    mae: bool = False,
    min_photons: int = 1,
    max_photons: int = 10000000,
    dist: str = '/',
    max_amplitude: float = 1.,
    embedding: str = 'spatial_planes',
    samplelimit: int = None,
    profile: bool = False,
    workers: int = 1,
    gpu_workers: int = 1,
    cpu_workers: int = 1,
):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger.info(f"Output dir: {outdir}")
    outdir.mkdir(exist_ok=True, parents=True)
    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)
    checkpointdir = outdir / 'checkpoints'
    checkpointdir.mkdir(exist_ok=True, parents=True)


    patches = [int(i) for i in patches.split('-')] if isinstance(patches, str) else patches
    heads = [int(i) for i in heads.split('-')] if isinstance(heads, str) else heads
    repeats = [int(i) for i in repeats.split('-')] if isinstance(repeats, str) else repeats

    network = network.lower()
    opt = opt.lower()

    if network == 'realspace':
        inputs = (batch_size, input_shape, input_shape, input_shape, 1)
    else:
        inputs = (batch_size, 6, input_shape, input_shape, 1)

    pmodes = modes if pmodes is None else pmodes

    if gpu_workers == -1:
        gpu_workers = torch.cuda.device_count()

    worker_batch_size = batch_size // (workers * gpu_workers)

    deepspeed_config = {
        "fp16": {
            "enabled": True if amp == 'fp16' else False,
            "auto_cast": True if amp == 'fp16' else False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": True,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": True if amp == 'bf16' else False,
            "auto_cast": True if amp == 'bf16' else False,
        },
        "zero_optimization": {
            "stage": 3,
            "reduce_bucket_size": "auto",
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            # HZP++ CONFIG
            # "zero_quantized_weights": True,
            # "zero_hpz_partition_size": 8, # Should not be set for single-node training, otherwise `gpus per node`
            # "zero_quantized_gradients": True,
            # OFFLOAD CONFIG
            "offload_optimizer": {
                "device": "none",
            },
            "offload_param": {
                "device": "none",
            },
        },
        "tensorboard": {
            "enabled": True,
            "output_path": str(logdir),
            "job_name": str(outdir.name)
        },
        "csv_monitor": {
            "enabled": True,
            "output_path": str(logdir),
            "job_name": str(outdir.name)
        },
        "gradient_clipping": clip_grad,
        "steps_per_print": 10,
        "gradient_accumulation_steps": 1,
        "train_batch_size": batch_size,
        # train_batch_size must be equal to train_micro_batch_size_per_gpu x gradient_accumulation_steps x GPUs.
        # "train_micro_batch_size_per_gpu": batch_size // (workers * gpu_workers),
        "zero_allow_untested_optimizer": True,
        "flops_profiler": {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": str(logdir / "flops_profiler.log")
        }
    }

    train_loop_config = {
        "epochs": epochs,
        "opt": opt,
        "lr": lr,
        "wd": wd,
        "ld": ld,
        "ema": ema,
        "clip_grad": clip_grad,
        "fixedlr": fixedlr,
        "warmup": warmup,
        "cooldown": cooldown,
        "dataset_size": None,
        "batch_size": worker_batch_size,
        "pmodes": pmodes,
        "inputs": inputs,
        "network": network,
        "distribution": dist,
        "embedding": embedding,
        "max_amplitude": max_amplitude,
        "samplelimit": samplelimit,
        "min_photons": min_photons,
        "max_photons": max_photons,
        "finetune": finetune,
        "dataset": dataset,
        "outdir": outdir,
        "logdir": logdir,
        "checkpointdir": checkpointdir,
        "cpu_workers": cpu_workers,
        "gpu_workers": gpu_workers,
        "workers": workers,
        "amp": amp,
        "repeats": repeats,
        "heads": heads,
        "patches": patches,
        "hidden_size": hidden_size,
        "profile": profile,
        "dropout": dropout,
        "fixed_dropout_depth": fixed_dropout_depth,
        "deepspeed_config": deepspeed_config,
    }

    scaling_config = ScalingConfig(
        num_workers=workers * gpu_workers,
        resources_per_worker={"CPU": cpu_workers // gpu_workers, "GPU": 1},
        trainer_resources={"CPU": 1},  # 1 cpu core for the training coordinator
        use_gpu=True
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute='loss',
        checkpoint_score_order='min',
    )

    run_config = RunConfig(
        log_to_file=True,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=outdir,
    )

    torch_config = TorchConfig(timeout_s=3600)

    if mae:
        training_paradigm = backend.pixel_reconstruction
    elif jepa:
        training_paradigm = backend.joint_embedding_prediction
    else:
        training_paradigm = backend.supervised

    trainer = TorchTrainer(
        train_loop_per_worker=training_paradigm,
        train_loop_config=train_loop_config,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None,
    )

    try:
        result = trainer.fit()
        logger.info(f"Model saved to {result.path}, {result.checkpoint}")
        logger.info(f"Training completed with metrics: {result.metrics}")
        logger.info(f"Error logs: {result.error}")
        logger.info(f"Best model checkpoint: {result.best_checkpoints}")

    except Exception as e:
        logger.info(f"Training failed with exception: {e}")
        sys.exit(1)


def main(args=None):

    # mp.set_start_method('spawn', force=True)

    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    try:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]
        # address = '127.0.1.1'
        # port = '32032'

        logger.info(f"Connecting to address: {address}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"}
        )

    except KeyError:
        logger.info(f"Starting a new local ray cluster")
        init(
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
            num_cpus=args.cpu_workers + 1,  # 1 cpu core for the training coordinator
            num_gpus=args.gpu_workers,
            ignore_reinit_error=True
        )

    logger.info('\nResources available to this Ray cluster:')
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')


    train_model(
        network=args.network,
        dataset=args.dataset,
        outdir=args.outdir,
        input_shape=args.input_shape,
        modes=args.modes,
        pmodes=args.pmodes,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        patches=args.patches,
        heads=args.heads,
        repeats=args.repeats,
        opt=args.opt,
        lr=args.lr,
        wd=args.wd,
        ld=args.ld,
        ema=args.ema,
        warmup=args.warmup,
        cooldown=args.cooldown,
        clip_grad=args.clip_grad,
        epochs=args.epochs,
        fixedlr=args.fixedlr,
        dropout=args.dropout,
        fixed_dropout_depth=args.fixed_dropout_depth,
        amp=args.amp,
        finetune=args.finetune,
        jepa=args.jepa,
        mae=args.mae,
        min_photons=args.min_photons,
        max_photons=args.max_photons,
        dist=args.dist,
        max_amplitude=args.max_amplitude,
        embedding=args.embedding,
        samplelimit=args.samplelimit,
        profile=args.profile,
        workers=args.workers,
        gpu_workers=args.gpu_workers,
        cpu_workers=args.cpu_workers,
    )

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")
    sys.exit(0)


if __name__ == "__main__":
    main()
