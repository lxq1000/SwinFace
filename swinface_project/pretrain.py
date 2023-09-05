import argparse
import logging
import os
from itertools import cycle

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import build_scheduler
from partial_fc import PartialFC, PartialFCAdamW
from analysis import *

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

    # Backbone
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    # Analysis subnets
    age_subnet = Subnet(in_chans=768, num_features=512, out_num=2, out_features=[1, 2]).cuda()
    gender_subnet = Subnet(in_chans=768, num_features=512, out_num=1, out_features=[2]).cuda()
    sd_subnet = Subnet(in_chans=768, num_features=512, out_num=8, out_features=[2 for j in range(8)]).cuda()
    hair_subnet = Subnet(in_chans=768, num_features=512, out_num=10, out_features=[2 for j in range(10)]).cuda()
    eyes_subnet = Subnet(in_chans=768, num_features=512, out_num=4, out_features=[2 for j in range(4)]).cuda()
    lower_subnet = Subnet(in_chans=768, num_features=512, out_num=11, out_features=[2 for j in range(11)]).cuda()
    whole_subnet = Subnet(in_chans=768, num_features=512, out_num=4, out_features=[2 for j in range(4)]).cuda()
    expression_subnet = Subnet(in_chans=768, num_features=512, out_num=2, out_features=[7, 2]).cuda()

    age_subnet = torch.nn.parallel.DistributedDataParallel(
        module=age_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    age_subnet.train()
    #age_subnet._set_static_graph()

    gender_subnet = torch.nn.parallel.DistributedDataParallel(
        module=gender_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    gender_subnet.train()
    #gender_subnet._set_static_graph()

    sd_subnet = torch.nn.parallel.DistributedDataParallel(
        module=sd_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    sd_subnet.train()
    #sd_subnet._set_static_graph()

    hair_subnet = torch.nn.parallel.DistributedDataParallel(
        module=hair_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    hair_subnet.train()
    #hair_subnet._set_static_graph()

    eyes_subnet = torch.nn.parallel.DistributedDataParallel(
        module=eyes_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    eyes_subnet.train()
    #eyes_subnet._set_static_graph()

    lower_subnet = torch.nn.parallel.DistributedDataParallel(
        module=lower_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    lower_subnet.train()
    #lower_subnet._set_static_graph()

    whole_subnet = torch.nn.parallel.DistributedDataParallel(
        module=whole_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    whole_subnet.train()
    #whole_subnet._set_static_graph()

    expression_subnet = torch.nn.parallel.DistributedDataParallel(
        module=expression_subnet, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    expression_subnet.train()
    #expression_subnet._set_static_graph()

    cfg.total_recognition_bz = cfg.recognition_bz * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_recognition_bz * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_recognition_bz * cfg.num_epoch

    cfg.total_batch_size = world_size * (cfg.recognition_bz + cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)

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
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": age_subnet.parameters()},
                    {"params": gender_subnet.parameters()},
                    {"params": sd_subnet.parameters()},
                    {"params": hair_subnet.parameters()},
                    {"params": eyes_subnet.parameters()},
                    {"params": lower_subnet.parameters()},
                    {"params": whole_subnet.parameters()},
                    {"params": expression_subnet.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": age_subnet.parameters()},
                    {"params": gender_subnet.parameters()},
                    {"params": sd_subnet.parameters()},
                    {"params": hair_subnet.parameters()},
                    {"params": eyes_subnet.parameters()},
                    {"params": lower_subnet.parameters()},
                    {"params": whole_subnet.parameters()},
                    {"params": expression_subnet.parameters()},
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
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])  # only load backbone!
        del dict_checkpoint

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_epoch_{cfg.resume_epoch}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        age_subnet.module.load_state_dict(dict_checkpoint["state_dict_age_subnet"])
        gender_subnet.module.load_state_dict(dict_checkpoint["state_dict_gender_subnet"])
        sd_subnet.module.load_state_dict(dict_checkpoint["state_dict_sd_subnet"])
        hair_subnet.module.load_state_dict(dict_checkpoint["state_dict_hair_subnet"])
        eyes_subnet.module.load_state_dict(dict_checkpoint["state_dict_eyes_subnet"])
        lower_subnet.module.load_state_dict(dict_checkpoint["state_dict_lower_subnet"])
        whole_subnet.module.load_state_dict(dict_checkpoint["state_dict_whole_subnet"])
        expression_subnet.module.load_state_dict(dict_checkpoint["state_dict_expression_subnet"])
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

    #with torch.no_grad():
    #    callback_verification(global_step, backbone)
    #    FGNet_verification(global_step, backbone, age_subnet)
    #    LAP_verification(global_step, backbone, age_subnet)
    #    CelebA_verification(global_step, backbone, age_subnet, gender_subnet, sd_subnet, hair_subnet, eyes_subnet,
    #                        lower_subnet, whole_subnet, expression_subnet)
    #    RAF_verification(global_step, backbone, expression_subnet)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        # print(len(train_loader),len(age_gender_train_loader),len(CelebA_train_loader),len(RAF_train_loader))

        for _, data in enumerate(
                zip(train_loader, age_gender_train_loader, CelebA_train_loader, Expression_train_loader)):
            global_step += 1

            # logging.info(str(global_step))

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

            # print(recognition_label, age_label, gender_label, smile_label, expression_label)

            img = torch.cat([recognition_img, age_gender_img, CelebA_img, expression_img], dim=0).cuda(
                non_blocking=True)

            # Concat images from different dataloaders
            local_features, global_features, x = backbone.module.forward_features(img)

            recognition_features = x[features_cut[0]: features_cut[1]]
            age_features = global_features[features_cut[1]: features_cut[2]]
            gender_features = global_features[features_cut[1]: features_cut[3]]
            CelebA_features = global_features[features_cut[2]: features_cut[3]]
            expression_features = global_features[features_cut[3]: features_cut[4]]

            local_embeddings = backbone.module.feature(recognition_features)

            [age, _] = age_subnet(age_features)
            [_, young] = age_subnet(CelebA_features)
            [gender] = gender_subnet(gender_features)
            [oval_face, pale_skin, big_lips, narrow_eyes, big_nose, pointy_nose, high_cheekbones,
             rosy_cheeks] = sd_subnet(CelebA_features)
            [bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline, straight_hair, wavy_hair,
             wearing_hat] = hair_subnet(CelebA_features)
            [arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses] = eyes_subnet(CelebA_features)
            [wearing_earrings, sideburns, five_o_clock_shadow, mouth_slightly_open, mustache, wearing_lipstick,
             no_beard, double_chin, goatee, wearing_necklace, wearing_necktie] = lower_subnet(CelebA_features)
            [attractive, blurry, chubby, heavy_makeup] = whole_subnet(CelebA_features)
            [expression, _] = expression_subnet(expression_features)
            [_, smiling] = expression_subnet(CelebA_features)

            analysis_outputs = [age, attractive, blurry, chubby, heavy_makeup, gender, oval_face, pale_skin, smiling,
                                young,
                                bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline,
                                straight_hair, wavy_hair, wearing_hat,
                                arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses, narrow_eyes, big_nose,
                                pointy_nose, high_cheekbones, rosy_cheeks, wearing_earrings,
                                sideburns, five_o_clock_shadow, big_lips, mouth_slightly_open, mustache,
                                wearing_lipstick, no_beard, double_chin, goatee, wearing_necklace,
                                wearing_necktie, expression]  # Total:42

            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)

            analysis_losses = []

            for j in range(42):
                if j == 0:  # age
                    
                    analysis_loss = criteria[j](analysis_outputs[j], analysis_labels[j], global_step)
                else:
                    analysis_loss = criteria[j](analysis_outputs[j], analysis_labels[j])
                analysis_losses.append(analysis_loss)

            loss = cfg.recognition_loss_weight * recognition_loss

            for j in range(42):
                loss += analysis_losses[j] * cfg.analysis_loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step - 1)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                for j in range(42):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                callback_logging(global_step, loss_am, recognition_loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                 lr_scheduler.get_update_values(global_step - 1)[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)
                    FGNet_verification(global_step, backbone, age_subnet)
                    LAP_verification(global_step, backbone, age_subnet)
                    CelebA_verification(global_step, backbone, age_subnet, gender_subnet, sd_subnet, hair_subnet,
                                        eyes_subnet, lower_subnet, whole_subnet, expression_subnet)
                    RAF_verification(global_step, backbone, expression_subnet)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_dict_age_subnet": age_subnet.module.state_dict(),
                "state_dict_gender_subnet": gender_subnet.module.state_dict(),
                "state_dict_sd_subnet": sd_subnet.module.state_dict(),
                "state_dict_hair_subnet": hair_subnet.module.state_dict(),
                "state_dict_eyes_subnet": eyes_subnet.module.state_dict(),
                "state_dict_lower_subnet": lower_subnet.module.state_dict(),
                "state_dict_whole_subnet": whole_subnet.module.state_dict(),
                "state_dict_expression_subnet": expression_subnet.module.state_dict(),

                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_epoch_{epoch}_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    with torch.no_grad():
        callback_verification(global_step, backbone)
        FGNet_verification(global_step, backbone, age_subnet)
        LAP_verification(global_step, backbone, age_subnet)
        CelebA_verification(global_step, backbone, age_subnet, gender_subnet, sd_subnet, hair_subnet, eyes_subnet,
                            lower_subnet, whole_subnet, expression_subnet)
        RAF_verification(global_step, backbone, expression_subnet)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
