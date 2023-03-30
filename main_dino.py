# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from segdino_model import DINOHead, SegDINO, Neck
from LoveDA import LoveDAdataset, DataAugmentationDINO
from segdino_loss import DINOLoss, init_msn_loss, AllReduceSum

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('SegDINO', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")


    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""0.0005 Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-8, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help="default=0.1,stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    parser.add_argument('--num_proto', default=1024, type=int, help="""K learnable prototypes""")
    parser.add_argument('--freeze_proto', default=False, type=bool, help="""whether to freeze prototypes or not""")

    """ target==>global;anchor==>local"""
    parser.add_argument('--target_view_crops_scale', type=float, nargs='+', default=(0.25, 0.45))
    parser.add_argument('--target_view_crops_size', type=int, nargs='+', default=(224, 224))
    parser.add_argument('--anchor_view_crops_scale', type=float, nargs='+', default=(0.25, 0.45))
    parser.add_argument('--anchor_view_crops_size', type=float, nargs='+', default=(96, 96))
    parser.add_argument('--focal_views_number', type=int, default=1, help="""Number of focal_views to generate. """)
    parser.add_argument('--local_crops_number', type=int, default=1, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)

    parser.add_argument('--mask_ratio', type=float, default=0.75, help="mask rate of target features")

    ## loss
    parser.add_argument('--label_smoothing', default=0.0, type=float, help="label_smoothing for prototypes")
    parser.add_argument('--loss_temperature', default=0.1, type=float, help="cosine similarity temperature")
    parser.add_argument('--use_sinkhorn', default=False, type=bool, help="sinkhorn to find the bast matching")
    parser.add_argument('--use_ent', default=False, type=bool, help="default=True")
    parser.add_argument('--ent_weight', default=0.0, type=float, help="default=0  cosine similarity temperature")
    parser.add_argument('--me_max', default=True, type=bool, help="sinkhorn to find the bast matching")
    parser.add_argument('--memax_weight', default=0.0, type=float, help="default= 1.0;cosine similarity temperature")
    parser.add_argument('--ploss_weight', default=1.0, type=float, help="default= 1.0;")

    # Misc
    parser.add_argument('--data_path', default='/media/data1/wjy/dataset/loveda/Test/',
                        type=str,help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./exps/0323/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--batch_size_per_gpu', default=48, type=int,
                        help='48  Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def create_logger(arg):
    # set up logger dir
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)

    final_log_file = os.path.join(arg.output_dir, log_file)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), filemode="a",
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def one_hot(targets, num_classes, smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1).cuda()
    return torch.full((len(targets), num_classes), off_value, device='cuda').scatter_(1, targets, on_value)


def train_dino(args):
    logger = create_logger(args)

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO()
    target_view = transforms.RandomResizedCrop(size=args.target_view_crops_size, scale=args.target_view_crops_scale)
    anchor_view = transforms.RandomResizedCrop(size=args.anchor_view_crops_size, scale=args.anchor_view_crops_scale)
    dataset = LoveDAdataset(target_view=target_view, anchor_view=anchor_view, transform=transform,
                            datadir=args.data_path, args=args)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    output_dim = 384  # default for ViT-S
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        output_dim = student.embed_dim  # output_dim from model
    else:
        logger.error(f"Unknow architecture: {args.arch}")

    neck = Neck(input_dim=output_dim, decoder_emd_dim=256, cross_atte_out_dim=128, mask_ratio=args.mask_ratio)

    # move networks to gpu
    student, teacher, neck = student.cuda(), teacher.cuda(), neck.cuda()
    # synchronize batch norms (if any)
    # if utils.has_batchnorms(student):
    #     student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    #     teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    #
    #     # we need DDP wrapper to have synchro batch norms working...
    #     teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    #     teacher_without_ddp = teacher.module
    # else:
    #     # teacher_without_ddp and teacher are the same thing
    #     teacher_without_ddp = teacher

    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())# teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Student and Teacher are built: they are both {args.arch} network.")

    # multi-crop wrapper handles forward with inputs of different resolutions
    segdino = SegDINO(student, teacher, neck)

    # -- make prototypes
    prototypes, proto_labels = None, None
    if args.num_proto > 0:
        with torch.no_grad():
            prototypes = torch.empty(args.num_proto, output_dim)
            _sqrt_k = (1. / output_dim) ** 0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)  # 从均匀分布U(a,b)中生成值，填充输入的张量或变量。
            prototypes = torch.nn.parameter.Parameter(prototypes).cuda()
            # -- init prototype labels
            proto_labels = one_hot(torch.tensor([i for i in range(args.num_proto)]), args.num_proto,
                                   args.label_smoothing)

        if not args.freeze_proto:
            prototypes.requires_grad = True

    # ============ preparing loss ... ============
    celoss = nn.BCEWithLogitsLoss()
    # -- init feature pre losses
    msn = init_msn_loss(
        num_views=args.focal_views_number + args.local_crops_number,
        tau=args.loss_temperature,
        me_max=args.me_max,
        return_preds=True)

    # ============ preparing optimizer ... ============
    # params_groups = utils.get_params_groups(student)
    params_groups = utils.get_params_groups(segdino)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / args.batch_size_per_gpu,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    logger.info(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        prototypes=prototypes,
        fp16_scaler=fp16_scaler,
        # dino_loss=dino_loss,
    )

    start_epoch = to_restore["epoch"]
    
    start_time = time.time()
    logger.info("Starting SegDINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(logger, segdino,
                                      data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                      epoch, fp16_scaler, celoss, msn, proto_labels, prototypes, output_dim, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': segdino.student.state_dict(),
            'teacher': segdino.teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'prototypes': prototypes.data,
            'epoch': epoch + 1,
            'args': args,
            # 'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch+1) % args.saveckp_freq == 0 or (epoch+1) == args.epochs:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch+1:04}.pth'))
        log_stats = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()},
                     }
        if utils.is_main_process():
            # logger.info(log_stats)
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(logger, segdino, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, celoss, msn, proto_labels, prototypes, output_dim, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch+1, args.epochs)
    for it, (target, anchors, mask_labels, img_mate) in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        anchors = [im.cuda(non_blocking=True) for im in anchors]
        mask_labels = [ t.cuda(non_blocking=True).flatten(1) for t in mask_labels ]  # [B, 64, 64] to [B, 4096] laction mask ground true

        ploss_meter = AverageMeter()
        rloss_meter = AverageMeter()
        eloss_meter = AverageMeter()
        BCEloss_meter = AverageMeter()

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            pre_mask_list, student_output_2mlp_list, teacher_output_2mlp = segdino(target, anchors)
            for pre_id in range(len(pre_mask_list)):
                BCEloss = celoss(pre_mask_list[pre_id], mask_labels[pre_id].float())
                BCEloss_meter.update(BCEloss)

            # Step 3. compute patch-based feature loss with me-max regularization. the features without via two mlp
            # teacher_output = teacher_output_2mlp.float().detach() # 32,198,384
            teacher_output = teacher_output_2mlp.float()# 32,198,384
            teacher_output_resized_list = []
            for j in range(len(pre_mask_list)): # number of crops
                teacher_output_resized = []
                for i in range(teacher_output.size()[0]): # number of batch
                    X = img_mate['croped_from'][j][0][i]
                    Y = img_mate['croped_from'][j][1][i]
                    W = img_mate['croped_from'][j][2][i]
                    H = img_mate['croped_from'][j][3][i]
                    # print("teacher_output[i].size():", teacher_output[i].size()) #[1024, 384]
                    # tem = teacher_output[i].transpose(1,0).reshape(-1,32,32)[:, Y:Y+H, X:X+W].unsqueeze(0)
                    tem = teacher_output[i].transpose(1, 0).reshape(-1, 14, 14)[:, Y:Y + H, X:X + W].unsqueeze(0)
                    # # [1024, 384]=>[384, 1024]=>[384, 32, 32]=>croped:[384, H, W] => [1, 384, H, W]
                    # print("tem.size()",tem.size())# [1, 384, H, W]
                    # tem = nn.functional.interpolate(tem, size=[14, 14], mode="bilinear") # [1, 384, H, W] => [1, 384, 14, 14]
                    tem = nn.functional.interpolate(tem, size=[6, 6], mode="bilinear")  # [1, 384, H, W] => [1, 384, 14, 14]
                    teacher_output_resized.append(tem)
                teacher_output_resized = torch.stack(teacher_output_resized, dim=0)
                teacher_output_resized = teacher_output_resized.reshape(args.batch_size_per_gpu, 1, output_dim, -1) \
                    .squeeze(1).transpose(2,1)  # target_croped size: [B, 384, 14, 14] =>[B, 1, 384, 196] =>[B, 384, 196] =>[B, 196, 384]
                teacher_output_resized_list.append(teacher_output_resized)

            for stu_out_id in range(len(student_output_2mlp_list)): # number of crops
                (ploss, res_me_max, res_ent, logs, _) = msn(
                    use_sinkhorn=args.use_sinkhorn,
                    use_entropy=args.use_ent,
                    anchor_views=student_output_2mlp_list[stu_out_id].float(),  # [B, 196, 384] # after a mlp head
                    target_views=teacher_output_resized_list[stu_out_id],  # [B, 196, 384] # without a mlp head
                    proto_labels=proto_labels,
                    prototypes=prototypes)
                ploss_meter.update(ploss)
                rloss_meter.update(res_me_max)
                eloss_meter.update(res_ent)

            loss = BCEloss_meter.avg + args.ploss_weight*ploss_meter.avg + args.memax_weight * rloss_meter.avg + args.ent_weight * eloss_meter.avg

        if not math.isfinite(loss.item()):
            logger.error("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            # print("fp16_scaler is None")
            # for name, parms in segdino.named_parameters():
            #     if "neck" in name:
            #         print('-->name:', name)
            #         print('-->para:', parms)
            #         print('-->grad_requirs:', parms.requires_grad)
            #         print('-->grad_value:', parms.grad)
            #         print("="*50)
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(segdino.student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, segdino.student,
                                              args.freeze_last_layer)
            optimizer.step()
            # print("=============更新之后===========")
            # for name, parms in segdino.named_parameters():
            #     if "neck" in name:
            #         print('-->name:', name)
            #         print('-->para:', parms)
            #         print('-->grad_requirs:', parms.requires_grad)
            #         print('-->grad_value:', parms.grad)
            #         print("="*50)
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(segdino.student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, segdino.student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # exit(0)

        # EMA update for the teacher
        with torch.no_grad():
            prototypes.grad.data = AllReduceSum.apply(prototypes.grad.data)
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(segdino.student.module.parameters(), segdino.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(BCEloss=BCEloss_meter.avg, ploss=ploss_meter.avg, res_me_max=rloss_meter.avg,
                             res_ent=eloss_meter.avg)
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:"+str(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SegDINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
