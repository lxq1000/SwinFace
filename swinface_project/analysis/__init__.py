
from .losses import AgeLoss
from .verification import FGNetVerification, CelebAVerification, RAFVerification, LAPVerification
from .task_name import ANALYSIS_TASKS
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

import torch
import numpy as np
import torch.distributed as dist
import os


from typing import Iterable
from functools import partial
from torchvision import transforms
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

from .datasets import AgeGenderDataset, CelebADataset, RAFDataset, FGnetDataset, ExpressionDataset, LAPDataset
from .samplers import SubsetRandomSampler
from dataset import MXFaceDataset


def get_analysis_train_dataloader(data_choose, config, local_rank) -> Iterable:

    if data_choose == "recognition":
        batch_size = config.recognition_bz
        root_dir = config.rec
        dataset_train = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    if data_choose == "age_gender":
        batch_size = config.age_gender_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            interpolation=config.INTERPOLATION,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = AgeGenderDataset(config=config, dataset=config.age_gender_data_list, transform=transform)

    elif data_choose == "CelebA":
        batch_size = config.CelebA_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            interpolation=config.INTERPOLATION,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = CelebADataset(config=config, choose="train", transform=transform)

    elif data_choose == "expression":
        batch_size = config.expression_bz
        transform = create_transform(
            input_size=config.img_size,
            scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
            ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
            is_training=True,
            color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
            auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG_REPROB,
            re_mode=config.AUG_REMODE,
            re_count=config.AUG_RECOUNT,
            interpolation=config.INTERPOLATION,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
        dataset_train = ExpressionDataset(config=config, transform=transform)

    rank, world_size = get_dist_info()
    sampler_train = DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=config.train_num_workers,
        pin_memory=config.train_pin_memory,
        drop_last=True,
    )
    return data_loader_train



def get_mixup_fn(config):

    mixup_fn = None
    mixup_active = config.AUG_MIXUP > 0 or config.AUG_CUTMIX > 0. or config.AUG_CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG_MIXUP, cutmix_alpha=config.AUG_CUTMIX, cutmix_minmax=config.AUG_CUTMIX_MINMAX,
            prob=config.AUG_MIXUP_PROB, switch_prob=config.AUG_MIXUP_SWITCH_PROB, mode=config.AUG_MIXUP_MODE,
            label_smoothing=config.RAF_LABEL_SMOOTHING, num_classes=config.RAF_NUM_CLASSES)
    return mixup_fn


def get_analysis_val_dataloader(data_choose, config):

    if data_choose == "CelebA":
        dataset_val = CelebADataset(config=config, choose="test")
    elif data_choose == "LAP":
        dataset_val = LAPDataset(config=config, choose="test")
    elif data_choose == "FGNet":
        dataset_val = FGnetDataset(config=config, choose="all")
    elif data_choose == "RAF":
        dataset_val = RAFDataset(config=config, choose="test")

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=config.val_pin_memory,
        drop_last=False
    )

    return data_loader_val

