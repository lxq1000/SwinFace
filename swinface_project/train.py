import argparse
import logging
import os
from itertools import cycle
import math

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import build_scheduler
from partial_fc import PartialFC, PartialFCAdamW
from analysis import *
from analysis import subnets
from model import build_model

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # Recognition dataloader
    '''
    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )
    '''
    train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)
    # Analysis dataloaders
    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

    model = build_model(cfg).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    model._set_static_graph()

    cfg.total_batch_size = world_size * (cfg.recognition_bz + cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    cfg.epoch_step = len(train_loader)

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    #cfg.total_recognition_bz = cfg.recognition_bz * world_size
    #cfg.warmup_step = cfg.num_image // cfg.total_recognition_bz * cfg.warmup_epoch
    #cfg.total_step = cfg.num_image // cfg.total_recognition_bz * cfg.num_epoch

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Recognition loss
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    age_loss = AgeLoss(total_iter=cfg.total_step)

    # Analysis task_losses
    criteria = [age_loss]
    criteria.extend([torch.nn.CrossEntropyLoss() for j in range(41)])  # Total:42

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        dict_checkpoint = torch.load(os.path.join(cfg.init_model, f"start_{rank}.pt"))
        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"],
                                              strict=False)  # only load backbone!
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])                                  
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch = start_epoch+1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    FGNet_loader = get_analysis_val_dataloader(data_choose="FGNet", config=cfg)
    LAP_loader = get_analysis_val_dataloader(data_choose="LAP", config=cfg)
    CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    RAF_loader = get_analysis_val_dataloader(data_choose="RAF", config=cfg)

    FGNet_verification = FGNetVerification(data_loader=FGNet_loader, summary_writer=summary_writer)
    LAP_verification = LAPVerification(data_loader=LAP_loader, summary_writer=summary_writer)
    CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer)
    RAF_verification = RAFVerification(data_loader=RAF_loader, summary_writer=summary_writer)

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    bzs = [cfg.recognition_bz, cfg.age_gender_bz, cfg.CelebA_bz, cfg.expression_bz]

    features_cut = [0 for i in range(5)]
    for i in range(1, 5):
        features_cut[i] = features_cut[i - 1] + bzs[i - 1]

    '''
    with torch.no_grad():
        model.module.set_output_type("Recognition")
        callback_verification(global_step, model)
        model.module.set_output_type("Age")
        FGNet_verification(global_step, model)
        LAP_verification(global_step, model)
        model.module.set_output_type("Attribute")
        CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        RAF_verification(global_step, model)
    '''

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        for idx, data in enumerate(
                zip(train_loader, age_gender_train_loader, CelebA_train_loader, Expression_train_loader)):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            recognition = data[0]
            age_gender = data[1]
            CelebA = data[2]
            RAF = data[3]

            recognition_img, recognition_label = recognition
            age_gender_img, [age_label, gender_label_1] = age_gender
            CelebA_img, CelebA_label = CelebA
            expression_img, expression_label = RAF
            gender_label = torch.cat([gender_label_1, CelebA_label[4]])

            recognition_label = recognition_label.cuda(non_blocking=True)
            age_label = age_label.cuda(non_blocking=True)
            gender_label = gender_label.cuda(non_blocking=True)
            expression_label = expression_label.cuda(non_blocking=True)

            analysis_labels = [age_label]
            for j in range(40):
                analysis_labels.append(CelebA_label[j].cuda(non_blocking=True))
            analysis_labels[5] = gender_label
            analysis_labels.append(expression_label)

            img = torch.cat([recognition_img, age_gender_img, CelebA_img, expression_img], dim=0).cuda(
                non_blocking=True)

            # Concat images from different dataloaders
            model.module.set_output_type("List")
            outputs = model(img)

            local_embeddings = outputs[-1][features_cut[0]: features_cut[1]]
            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)

            analysis_losses = []

            for j in range(42):
                if j == 0:  # age
                    analysis_output = outputs[j][features_cut[1]: features_cut[2]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j], global_step)
                elif j == 5:
                    analysis_output = outputs[j][features_cut[1]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])
                elif j == 41:
                    analysis_output = outputs[j][features_cut[3]: features_cut[4]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])
                else:
                    analysis_output = outputs[j][features_cut[2]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])

                analysis_losses.append(analysis_loss)

            loss = cfg.recognition_loss_weight * recognition_loss

            for j in range(42):
                loss += analysis_losses[j] * cfg.analysis_loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                for j in range(42):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                callback_logging(global_step, loss_am, recognition_loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step)[0], amp)

                if (global_step+1) % cfg.verbose == 0:
                    model.module.set_output_type("Recognition")
                    callback_verification(global_step, model)
                    model.module.set_output_type("Age")
                    FGNet_verification(global_step, model)
                    LAP_verification(global_step, model)
                    model.module.set_output_type("Attribute")
                    CelebA_verification(global_step, model)
                    model.module.set_output_type("Expression")
                    RAF_verification(global_step, model)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            train_loader.reset()

    with torch.no_grad():
        model.module.set_output_type("Recognition")
        callback_verification(global_step, model)
        model.module.set_output_type("Age")
        FGNet_verification(global_step, model)
        LAP_verification(global_step, model)
        model.module.set_output_type("Attribute")
        CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        RAF_verification(global_step, model)

    # if rank == 0:
    # path_module = os.path.join(cfg.output, "model.pt")
    # torch.save(backbone.module.state_dict(), path_module)

    # from torch2onnx import convert_onnx
    # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
