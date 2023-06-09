import os
import logging
import random
import warnings
import numpy as np

import megengine
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.autodiff as autodiff
import megengine.distributed as dist
import megengine.data as data

from train_utils import TBLog, get_optimizer
from utils import get_logger, net_builder, str2bool, over_write_args_from_file
from models.fullmatch.fullmatch import FullMatch
from datasets.ssl_dataset import SSL_Dataset


def worker(args):

    args.world_size = dist.get_world_size()
    args.gpu = dist.get_rank()
    save_path = os.path.join(args.save_dir, args.save_name)

    if args.seed is not None:
        random.seed(args.seed)
        megengine.random.seed(args.seed)
        np.random.seed(args.seed)

    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.gpu == 0 :
        tb_log = TBLog(args.save_dir, args.save_name)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training") 

    if args.dataset.upper() == 'CIFAR100' and args.num_labels==400 and args.world_size > 1:
        args.sync_bn = True

    args.bn_momentum = 0.999
    if 'imagenet' in args.dataset.lower():
        print('Please Waiting for Supporting')
        exit()
    else:
        _net_builder = net_builder(args.net, args.net_from_name,
                                    {'first_stride': 2 if 'stl' in args.dataset else 1,
                                    'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'bn_momentum': args.bn_momentum,
                                    'dropRate': args.dropout,
                                    'use_embed': False,
                                    'is_remix': False,
                                    'sync_bn': args.sync_bn},)

    model = FullMatch(_net_builder, args.num_classes, args.ema_m, args.p_cutoff, args.ulb_loss_ratio, args.hard_label,
                        num_eval_iter=args.num_eval_iter, tb_log=tb_log, logger=logger)

    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    model.set_optimizer(optimizer)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        checkpoint = megengine.load(args.resume, map_location='cpu')
        model.model.load_state_dict(checkpoint['state_dict'])
        model.ema.ema.load_state_dict(checkpoint['ema_state_dict'])
        megengine.distributed.group_barrier()

    args.batch_size = int(args.batch_size / args.world_size)
    logger.info(f"model_arch: {model}")

    if args.dataset != "imagenet":
        if args.gpu != 0:
            megengine.distributed.group_barrier()
        train_dset = SSL_Dataset(args, name=args.dataset, train=True, num_classes=args.num_classes, data_dir=args.data_dir)
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)

        _eval_dset = SSL_Dataset(args, name=args.dataset, train=False, num_classes=args.num_classes, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()
        if args.gpu == 0:
            megengine.distributed.group_barrier()
    else:
        print('Please Waiting for Supporting')
        exit()

    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = data.DataLoader(dset_dict['train_lb'],
                                        sampler= data.Infinite(data.RandomSampler(dset_dict['train_lb'], batch_size=args.batch_size)),
                                        num_workers=args.num_workers)
    loader_dict['train_ulb'] = data.DataLoader(dset_dict['train_ulb'],
                                        sampler=data.Infinite(data.RandomSampler(dset_dict['train_ulb'], batch_size=args.batch_size*args.uratio)),
                                        num_workers=args.num_workers)
    loader_dict['eval'] = data.DataLoader(dset_dict['eval'],
                                    sampler=data.SequentialSampler(dset_dict['eval'], batch_size=args.eval_batch_size,),
                                    num_workers=0)

    model.set_data_loader(loader_dict)
    megengine.distributed.group_barrier()

    trainer = model.train
    trainer(args, logger=logger)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FullMatch Training')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fullmatch')
    parser.add_argument('--resume', type=str,default=None)

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2**20, help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=1024, help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7, help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='batch size of evaluation data loader')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--loss_warm', type=bool, default=False)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')

    '''
    multi-GPUs & Distrbitued Training
    '''
    parser.add_argument('-n','--ngpus', default=8, type=int,help='number of GPUs per node (default: None, use all available GPUs)',)
    parser.add_argument('--dist-addr', default='localhost')
    parser.add_argument('--dist-port', default=23456, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)


    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    args.cur_dir = os.getcwd().split('/')[-1]

    args.distributed = False

    if args.ngpus is None:
        args.ngpus = dist.helper.get_device_count_by_fork('gpu')

    if args.world_size * args.ngpus > 1:
        args.distributed = True
        dist_worker = dist.launcher(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=args.ngpus,
            rank_start=args.rank * args.ngpus,
            n_gpus=args.ngpus
        )(worker)
        dist_worker(args)
    else:
        worker(args)
