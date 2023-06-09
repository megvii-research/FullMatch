import numpy as np
import json
import os
import math

def split_ssl_data(args, data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, num_classes, index)
    if include_lb_to_ulb:
        ulb_idx = np.array(range(len(target)))
        unlabeled_expand = (args.batch_size*args.world_size*args.uratio*args.num_eval_iter)//len(target)
        if unlabeled_expand == 0:
            unlabeled_expand += 1
        ulb_idx = np.hstack([ulb_idx for _ in range(unlabeled_expand)])
        np.random.shuffle(ulb_idx)
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]
    else:
        ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
        unlabeled_expand = math.ceil(args.batch_size*args.num_eval_iter/len(target))
        ulb_idx = np.hstack([ulb_idx for _ in range(unlabeled_expand)])
        np.random.shuffle(ulb_idx)
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(args, data, target, num_labels, num_classes, index=None, name=None):
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

    lb_idx = np.array(lb_idx)
    labeled_expand = math.ceil(args.batch_size*args.world_size*args.num_eval_iter/num_labels)
    lb_idx = np.hstack([lb_idx for _ in range(labeled_expand)])
    np.random.shuffle(lb_idx)

    lb_data.extend(data[lb_idx])
    lbs.extend(target[lb_idx])

    return np.array(lb_data), np.array(lbs), lb_idx