from tensorboardX import SummaryWriter
import megengine as mge
import megengine.functional as F
import megengine.distributed as dist

from copy import deepcopy
import os 
import math


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    optimizer = mge.optimizer.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    
    return optimizer

def adjust_learning_rate(optimizer, current_step, num_training_steps, num_cycles=7. / 16., num_warmup_steps=0, base_lr=0.03):

    if current_step < num_warmup_steps:
        _lr = float(current_step) / float(max(1, num_warmup_steps))
    else:
        num_cos_steps = float(current_step - num_warmup_steps)
        num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
        _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
    _lr = _lr * base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = _lr
    return _lr
        
class EMA(object):
    def __init__(self, model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay

    def update(self, model):

        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema.named_parameters())

        assert model_params.keys() == ema_params.keys(), 'Model parameter keys incompatible with EMA stored parameter keys'

        for name, param in model_params.items():
            ema_params[name].set_value(F.mul(ema_params[name], self.decay))
            ema_params[name].set_value(F.add(ema_params[name], (1 - self.decay) * param.detach()))
        
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema.named_buffers()) 

        assert model_buffers.keys() == ema_buffers.keys(), 'Model parameter keys incompatible with EMA stored parameter keys'  
        for name, buffer in model_buffers.items():
            ema_buffers[name].set_value(buffer.detach())

class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None, mode="train"):
        if suffix is None:
            suffix = ''
        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix + key, value, it)
        self.writer.flush()
            
    def close(self):
        self.writer.close()

class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ce_loss(logits, targets, use_hard_labels=True, reduction='mean'):
    log_pred = F.logsoftmax(logits, axis=-1)
    loss = -F.gather(log_pred, 1, targets.reshape(-1,1))
    if reduction == 'none':
        return loss.reshape(-1)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()

def reduce_tensor(tensor, mean=True):
    ts = F.distributed.all_reduce_sum(tensor)
    if mean:
        return ts / dist.get_world_size()
    return ts




