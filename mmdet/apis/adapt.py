# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results, build_optimizer
from mmdet.utils import get_device


def single_gpu_adapt(model,
                    data_loader,
                    cfg,
                    wandb_init_idx=0,
                    wandb=None):
    optimizer = model.module.build_optimizer(cfg.optimizer)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    iter_idx = 0
    for i, data in enumerate(data_loader):
        outputs = model(rescale=True, **data)

        result = outputs['results']
        batch_size = len(result)
        iter_idx += batch_size
        if 'loss' in outputs:
            optimizer.zero_grad()
            sum(outputs['loss'].values()).backward()
            print({k: '{:.3f}'.format(outputs['loss'][k].item()) for k in outputs['loss']})
            wandb.log(outputs['loss'], step=wandb_init_idx + iter_idx)
            optimizer.step()

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results, wandb_init_idx + iter_idx

