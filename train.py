import os
import os.path as op
import torch
import numpy as np
import random
import time


from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from utils.wandb_utils import setup_wandb, wandb_finish


def set_seed(seed=0, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    else:
        # Preserve the historical dm-adapter runtime behavior unless deterministic mode is requested.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(False)


def log_reproducibility_settings(logger, args, effective_seed):
    matmul_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    deterministic_algorithms = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    logger.info("Seed: base=%d, rank=%d, effective=%d", args.seed, get_rank(), effective_seed)
    logger.info("Deterministic mode: %s", "enabled" if args.deterministic else "disabled")
    logger.info("cuDNN deterministic: %s", torch.backends.cudnn.deterministic)
    logger.info("cuDNN benchmark: %s", torch.backends.cudnn.benchmark)
    logger.info("TF32 matmul: %s, TF32 cuDNN: %s", matmul_tf32, cudnn_tf32)
    logger.info("torch deterministic algorithms: %s", deterministic_algorithms)

def _count_parameters(module, trainable_only=False):
    return sum(
        p.numel()
        for p in module.parameters()
        if not trainable_only or p.requires_grad
    )


def _count_buffers(module):
    return sum(buffer.numel() for buffer in module.buffers())


def log_model_parameter_counts(model, logger):
    total_params = _count_parameters(model)
    trainable_params = _count_parameters(model, trainable_only=True)
    logger.info('Total params: %2.fM' % (total_params / 1000000.0))
    logger.info(
        "Trainable params: %d total (%.4fM), %.2f%% of total model params",
        trainable_params,
        trainable_params / 1000000.0,
        (trainable_params / total_params * 100.0) if total_params else 0.0,
    )

    prototype_branch = getattr(model, "prototype_branch", None)
    if prototype_branch is None:
        logger.info("Prototype branch params: disabled")
        return

    prototype_params = _count_parameters(prototype_branch)
    prototype_trainable_params = _count_parameters(prototype_branch, trainable_only=True)
    prototype_buffer_elements = _count_buffers(prototype_branch)
    prototype_share = (prototype_params / total_params * 100.0) if total_params else 0.0
    logger.info(
        "Prototype branch params: %d total (%.4fM), %d trainable (%.4fM), %.2f%% of total model params",
        prototype_params,
        prototype_params / 1000000.0,
        prototype_trainable_params,
        prototype_trainable_params / 1000000.0,
        prototype_share,
    )
    logger.info(
        "Prototype branch buffers: %d elements (%.4fM, not counted as params)",
        prototype_buffer_elements,
        prototype_buffer_elements / 1000000.0,
    )
    
if __name__ == '__main__':

    args = get_args()
    effective_seed = args.seed + get_rank()
    set_seed(effective_seed, deterministic=args.deterministic)
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = args.run_time or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('dm-adapter', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    log_reproducibility_settings(logger, args, effective_seed)
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    wandb_run = setup_wandb(args, cur_time, logger)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)

    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
        
    # frezze clip
    name_no_update = ["base_model"]
    for name, param in model.named_parameters():
        if any(ntu in name for ntu in name_no_update):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    name_update = ["adapter_mlp", "ln_3", "experts", "feed_forward", "ln_4", "param", "v2i_proj", "task_param"]
    for name, param in model.named_parameters():
        if any(ntu in name for ntu in name_update):
            param.requires_grad_(True)

    # Double check
    enabled = set()
    disabled = set()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(param_name)
        else:
            disabled.add(param_name)
    logger.info("Parameters to be updated: %d tensors", len(enabled))
    logger.info("Parameters not to be updated: %d tensors", len(disabled))
    log_model_parameter_counts(model, logger)
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    try:
        do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
    finally:
        if wandb_run is not None:
            wandb_finish()
