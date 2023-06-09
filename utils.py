import os
import time
import logging
import yaml

import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml

import megengine as mge 

mge.Parameter


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    for key in kwargs.keys():
        # if hasattr(cls, key):
        #     print(f"{key} in {cls} : {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    
    if net_name == 'WideResNet':
        import models.nets.wrn as net
        builder = getattr(net, 'build_WideResNet')()
    else:
        assert Exception("Not Implemented Error")

    if net_name != 'ResNet50':
        setattr_cls_from_kwargs(builder, net_conf)
    return builder.build


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
