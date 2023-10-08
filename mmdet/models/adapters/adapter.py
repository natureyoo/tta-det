# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.visualization import imshow_det_bboxes
from mmdet.core import build_optimizer

from ..builder import ADAPTERS


@ADAPTERS.register_module()
class Adapter(BaseModule, metaclass=ABCMeta):

    def __init__(self, init_cfg=None,
                 is_adapt=True,
                 where=None,
                 how=None,
                 gamma=128,
                 source_stats=None,
                 detector=None):
        super(Adapter, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.is_adapt = is_adapt
        self.where = where
        self.how = how
        self.gamma = 1 / gamma
        self.detector = detector
        self.learnable_params = None
        if source_stats is not None:
            self.s_stats, self.t_stats, self.alpha = self.initialize_stats(source_stats)
        self.set_adaptable_params()

    @property
    def with_detector(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'detector') and self.detector is not None

    def initialize_stats(self, source_stats):
        s_stats = torch.load(source_stats)
        t_stats = {k: s_stats[k] for k in s_stats}
        alpha = {k: torch.eye(s_stats[k][2].shape[0]) * s_stats[k][3].max().item() / 30 for k in s_stats}
        return s_stats, t_stats, alpha

    def set_adaptable_params(self):
        self.detector.eval()
        self.detector.backbone.requires_grad_(False)
        self.detector.neck.requires_grad_(False)
        self.detector.rpn_head.requires_grad_(False)
        self.detector.roi_head.requires_grad_(False)
        if self.is_adapt:
            if self.where == 'full':
                self.detector.backbone.requires_grad_(True)
                self.learnable_params = list(self.detector.backbone.parameters())
            elif self.where == 'adapter':
                self.learnable_params = self.detector.backbone.set_adaptable_params()
        else:
            self.detector.eval()

    def build_optimizer(self, cfg):
        # optimizer = build_optimizer(self.learnable_params, cfg)
        if 'type' in cfg:
            cfg.pop('type')
        if 'paramwise_cfg' in cfg:
            cfg.pop('paramwise_cfg')
        optimizer = torch.optim.AdamW(self.learnable_params, **cfg)
        return optimizer

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def accumulate_feats(self, features):
        if self.features is None:
            self.features = {idx: torch.nn.AdaptiveAvgPool2d((2**(idx+1), 2**(idx+1)))(f.detach()) for idx, f in enumerate(features)}
        else:
            for idx, f in enumerate(features):
                self.features[idx] = torch.cat([self.features[idx], torch.nn.AdaptiveAvgPool2d((2**(idx+1), 2**(idx+1)))(f.detach())], dim=0)

    def forward_train(self, imgs, img_metas, **kwargs):
        outputs = {}
        if self.is_adapt:
            feats, results = self.detector.adapt(imgs, img_metas, **kwargs)
            outputs['loss'] = self.loss(feats)
        else:
            results = self.detector.forward_test(imgs, img_metas, **kwargs)

        outputs['results'] = results
        return outputs

    def loss(self, feats):
        losses = {}
        if self.how == 'ema-kl':
            losses['ema-kl'] = 0
            for idx, _f in enumerate(feats):
                f = _f.mean(dim=[2,3])
                diff = f - self.t_stats[idx][2][None, :].to(f.device)
                delta = self.gamma * diff.sum(dim=0)
                cur_t_mean = self.t_stats[idx][2].to(f.device) + delta
                cur_t_cov = self.t_stats[idx][3].to(f.device) \
                                 + self.gamma * (diff.t() @ diff - self.t_stats[idx][3].to(f.device) * f.shape[0])\
                                 - delta.reshape(-1, 1) @ delta.reshape(1, -1)
                t_dist = torch.distributions.MultivariateNormal(cur_t_mean, cur_t_cov + self.alpha[idx].to(f.device))
                s_dist = torch.distributions.MultivariateNormal(self.s_stats[idx][2].to(f.device), self.s_stats[idx][3].to(f.device))
                self.t_stats[idx] = (self.t_stats[idx][0], self.t_stats[idx][1], cur_t_mean.detach(), cur_t_cov.detach())
                losses['ema-kl'] += (torch.distributions.kl.kl_divergence(s_dist, t_dist)
                               + torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2
        elif self.how == 'ema-l1':
            losses['ema-l1-mean'], losses['ema-l1-var'] = 0, 0
            for idx, _f in enumerate(feats):
                f = torch.nn.AdaptiveAvgPool2d((2 ** (idx + 1), 2 ** (idx + 1)))(_f)
                cur_t_mean = (1 - 1 / self.gamma) * self.t_stats[idx][0].to(f.device) + 1 / self.gamma * f.mean(dim=0)
                cur_t_var = (1 - 1 / self.gamma) * self.t_stats[idx][1].to(f.device) + 1 / self.gamma * f.var(dim=0)
                self.t_stats[idx] = (cur_t_mean.detach(), cur_t_var.detach(), self.t_stats[idx][2], self.t_stats[idx][3])
                losses['ema-l1-mean'] += torch.nn.L1Loss()(self.s_stats[idx][0].to(f.device), cur_t_mean)
                losses['ema-l1-var'] += torch.nn.L1Loss()(self.s_stats[idx][1].to(f.device), cur_t_var)
        elif self.how == 'ema-l1-mean':
            losses['ema-l1-mean'] = 0
            for idx, _f in enumerate(feats):
                f = torch.nn.AdaptiveAvgPool2d((2 ** (idx + 1), 2 ** (idx + 1)))(_f)
                cur_t_mean = (1 - 1 / self.gamma) * self.t_stats[idx][0].to(f.device) + 1 / self.gamma * f.mean(dim=0)
                self.t_stats[idx] = (cur_t_mean.detach(), self.t_stats[idx][1], self.t_stats[idx][2], self.t_stats[idx][3])
                losses['ema-l1-mean'] += torch.nn.L1Loss()(self.s_stats[idx][0].to(f.device), cur_t_mean)
        elif self.how == 'cur-l1':
            losses['cur-l1-mean'], losses['cur-l1-var'] = 0, 0
            for idx, _f in enumerate(feats):
                f = torch.nn.AdaptiveAvgPool2d((2 ** (idx + 1), 2 ** (idx + 1)))(_f)
                losses['cur-l1-mean'] += torch.nn.L1Loss()(self.s_stats[idx][0].to(f.device), f.mean(dim=0))
                losses['cur-l1-var'] += torch.nn.L1Loss()(self.s_stats[idx][1].to(f.device), f.var(dim=0))
        elif self.how == 'cur-mean-l1':
            losses['cur-l1-mean'] = 0
            for idx, _f in enumerate(feats):
                f = torch.nn.AdaptiveAvgPool2d((2 ** (idx + 1), 2 ** (idx + 1)))(_f)
                losses['cur-l1-mean'] += torch.nn.L1Loss()(self.s_stats[idx][0].to(f.device), f.mean(dim=0))

        return losses



    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss, log_vars=log_vars_, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    show_box_only=False,
                    show_mask_only=False):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            show_box_only=show_box_only,
            show_mask_only=show_mask_only)

        if not (show or out_file):
            return img

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
