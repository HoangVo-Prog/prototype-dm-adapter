import logging
import time
import torch
from torch.utils.data import DataLoader
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.train_diagnostics import compute_train_diagnostics
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
import copy
from pynvml import *
from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _prototype_requested(args):
    return (
        getattr(args, "prototype", False)
        or getattr(args, "use_loss_id", False)
    )


def _prototype_ready(model):
    model = _unwrap_model(model)
    branch = getattr(model, "prototype_branch", None)
    return branch is not None and branch.is_ready()


def _set_epoch_on_loader(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    batch_sampler = getattr(loader, "batch_sampler", None)
    inner_sampler = getattr(batch_sampler, "sampler", None)
    if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
        inner_sampler.set_epoch(epoch)


def _build_prototype_init_loader(train_loader, args):
    train_set = getattr(train_loader, "dataset", None)
    source_dataset = getattr(train_set, "dataset", None)
    if train_set is None or source_dataset is None:
        return None

    prototype_set = ImageTextDataset(
        source_dataset,
        transform=build_transforms(img_size=args.img_size, aug=False, is_train=False),
        text_length=getattr(train_set, "text_length", args.text_length),
        truncate=getattr(train_set, "truncate", True),
    )

    return DataLoader(
        prototype_set,
        batch_size=getattr(args, "test_batch_size", args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )


@torch.no_grad()
def _project_prototype_feature_bank(branch, image_features, text_features, batch_size):
    image_projected, text_projected = [], []
    batch_size = max(int(batch_size), 1)
    for start in range(0, image_features.shape[0], batch_size):
        end = start + batch_size
        image_batch, text_batch = branch.project_for_memory(
            image_features[start:end],
            text_features[start:end],
        )
        image_projected.append(image_batch.cpu())
        text_projected.append(text_batch.cpu())
    return torch.cat(image_projected, dim=0), torch.cat(text_projected, dim=0)


def _validate_prototype_init_features(image_features, text_features, pids, expected_len, num_classes):
    if image_features.shape[0] != text_features.shape[0] or image_features.shape[0] != pids.numel():
        raise ValueError(
            "prototype initialization feature count mismatch: "
            f"image={image_features.shape[0]}, text={text_features.shape[0]}, pids={pids.numel()}"
        )
    if expected_len is not None and image_features.shape[0] != expected_len:
        raise ValueError(
            f"prototype initialization scanned {image_features.shape[0]} samples, expected {expected_len}"
        )
    if pids.numel() == 0:
        raise ValueError("prototype initialization saw no samples")
    if pids.min().item() < 0 or pids.max().item() >= num_classes:
        raise ValueError(
            f"prototype pids must be in [0, {num_classes - 1}], "
            f"got min={pids.min().item()} max={pids.max().item()}"
        )
    unique = torch.unique(pids.cpu().long(), sorted=True)
    expected = torch.arange(num_classes, dtype=torch.long)
    if unique.numel() != expected.numel() or not torch.equal(unique, expected):
        missing = sorted(set(expected.tolist()) - set(unique.tolist()))
        raise ValueError(
            "prototype initialization did not cover every train identity; "
            f"missing {len(missing)} identities, first missing={missing[:10]}"
        )


@torch.no_grad()
def maybe_initialize_prototypes(model, train_loader, args, device, logger):
    model_without_ddp = _unwrap_model(model)
    branch = getattr(model_without_ddp, "prototype_branch", None)
    if branch is None or branch.is_ready():
        return

    logger.info("Initializing PBT prototypes from full train embeddings")
    was_training = model_without_ddp.training
    prototype_loader = _build_prototype_init_loader(train_loader, args)
    if prototype_loader is not None:
        logger.info("Using a dedicated no-augmentation loader for prototype initialization")
    else:
        logger.warning("Falling back to the training loader for prototype initialization")
        prototype_loader = train_loader

    image_features, text_features, pids = [], [], []
    try:
        model_without_ddp.eval()
        for batch in prototype_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            image_feat, text_feat = model_without_ddp.extract_prototype_features(batch)
            image_features.append(image_feat.cpu())
            text_features.append(text_feat.cpu())
            pids.append(batch['pids'].cpu())
    finally:
        model_without_ddp.train(was_training)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    pids = torch.cat(pids, dim=0)
    dataset = getattr(prototype_loader, "dataset", None)
    expected_len = len(dataset) if dataset is not None else None
    _validate_prototype_init_features(
        image_features,
        text_features,
        pids,
        expected_len,
        branch.memory.num_classes,
    )

    if hasattr(branch, "needs_pca_init") and branch.needs_pca_init():
        logger.info("Initializing prototype projector from raw train embeddings")
        branch.initialize_projector_from_features(image_features, text_features)

    image_features, text_features = _project_prototype_feature_bank(
        branch,
        image_features,
        text_features,
        getattr(args, "test_batch_size", getattr(args, "batch_size", 512)),
    )
    branch.initialize_projected(image_features, text_features, pids)
    logger.info("Prototype banks initialized with %d samples", pids.numel())
    synchronize()


def _loss_components(ret):
    return {
        key: value
        for key, value in ret.items()
        if "loss" in key and torch.is_tensor(value)
    }


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    if torch.is_tensor(value):
        value = value.detach().item()
    meters[key].update(value, batch_size)


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("dm-adapter.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "imkt_loss": AverageMeter(),
        "triplet_loss": AverageMeter(),
        "triplet_enhance_loss": AverageMeter(),
        "triplet_enhance_shuffle_loss": AverageMeter(),
        "aux_loss": AverageMeter(),
        "proto_id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    train_diag_state = {"assignments": {}}

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        _set_epoch_on_loader(train_loader, epoch)
        if _prototype_requested(args):
            if epoch > getattr(args, "prototype_warmup_epochs", 0) and not _prototype_ready(model):
                maybe_initialize_prototypes(model, train_loader, args, device, logger)
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)

            loss_components = _loss_components(ret)
            total_loss = sum(loss_components.values())

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            for loss_key, loss_value in loss_components.items():
                _update_meter(meters, loss_key, loss_value, batch_size)

            for metric_key in ('img_acc', 'txt_acc', 'mlm_acc'):
                if metric_key in ret:
                    _update_meter(meters, metric_key, ret[metric_key], batch_size)

            if (n_iter + 1) % log_period == 0:
                train_diag_metrics = compute_train_diagnostics(model, ret, args, train_diag_state)
                for diag_key, diag_value in train_diag_metrics.items():
                    _update_meter(meters, diag_key, diag_value, batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    # checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
    nvmlShutdown()



def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("dm-adapter.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
