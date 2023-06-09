import megengine as mge
import megengine.functional as F
import numpy as np
from train_utils import ce_loss, reduce_tensor


def consistency_loss(logits_s, logits_w, name='ce', p_cutoff=0.95, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    assert name == 'ce', 'must ce'
    pseudo_label = F.softmax(logits_w, axis=-1)
    max_probs = F.max(pseudo_label, axis=-1)
    max_idx = F.argmax(pseudo_label, axis=-1)
    mask = F.greater_equal(max_probs, p_cutoff).astype('float32')
    select = F.greater_equal(max_probs, p_cutoff).astype('int32')
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        print('must use hard label')
    return masked_loss.mean(), mask.mean(), select, max_idx

def nl_em_loss(pred_s, pred_w, k, mask_pred, p_cutoff):
    softmax_pred = F.softmax(pred_s, axis=-1)
    pseudo_label = F.softmax(pred_w, axis=-1)
    topk = F.topk(pseudo_label, k)[1]
    mask_k = F.scatter(F.ones_like(pseudo_label), 1, topk, F.zeros_like(topk))
    mask_k_npl = F.where((mask_k==1)&(softmax_pred>p_cutoff**2), F.zeros_like(mask_k), mask_k)
    loss_npl = (-F.log(1-softmax_pred+1e-10) * mask_k_npl).sum(axis=1).mean()

    label = F.argmax(pseudo_label, axis=-1)
    mask_k = F.scatter(mask_k, 1, label.reshape(-1,1), F.ones_like(label.reshape(-1,1)))

    yg = F.cond_take(mask_k.astype('bool'), softmax_pred)[0].reshape(pred_w.shape[0],-1).sum(axis=-1,keepdims=True)
    soft_ml = F.broadcast_to((1-yg+1e-7)/(k-1), pred_s.shape) 
    mask = 1 - mask_k
    mask = mask * mask_pred.reshape(-1,1)
    mask = F.where((mask==1)&(softmax_pred>p_cutoff**2), F.zeros_like(mask), mask)
    loss_em = -(soft_ml*F.log(softmax_pred+1e-10)+(1-soft_ml)*F.log(1-softmax_pred+1e-10))
    loss_em = (loss_em * mask).sum()/(mask.sum()+1e-10)
    return loss_npl, loss_em
    
def cal_topK(pred_s, pred_w, topk=(1,)):
    target_w = F.argmax(pred_w, axis=-1)
    output = pred_s
    target = target_w

    maxk = max(topk)
    batch_size = target.size

    _, pred = F.topk(output, maxk)
    pred = F.transpose(pred, pattern=(1,0))
    correct =  F.equal(pred, F.broadcast_to(target.reshape(1,-1), pred.shape)).astype('float32') 

    for k in list(np.arange(topk[0], topk[1]+1)):
        correct_k = correct[:k].reshape(-1).sum(0)
        acc_single = F.mul(correct_k, 100./batch_size)
        acc_parallel = reduce_tensor(acc_single)
        if acc_parallel > 99.99:
            return k
            
