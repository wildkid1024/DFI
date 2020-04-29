import time
import torch
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen
from lib.data import get_split_dataset
import math

import numpy as np
import copy

class Pruner:
    def __init__(self, prune_layer, prune_strategy, preserve_idx=None, channel_round=8):
        self.prune_layer = prune_layer
        self.prune_strategy = prune_strategy

    def prune_kernels(self, model):
        m_list = list(self.model.modules())
        for i, p in enumerate(self.prune_strategy): 
            idx = self.prune_layer[i]
            self.prune_kernel_per_layer(m_list[idx], p)

    def prune_kernel_per_layer(self, layer, preserve_ratio, preserve_idx=None):
        '''Return the real ratio'''
        op = layer
        assert (preserve_ratio <= 1.)

        if preserve_ratio == 1:  # do not prune
            return 1., op.weight.size(1), None  # TODO: should be a full index
            # n, c, h, w = op.weight.size()
            # mask = np.ones([c], dtype=bool)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        n, c = op.weight.size(0), op.weight.size(1)
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        extract_t1 = time.time()
        if self.use_new_input:  # this is slow and may lead to overfitting
            self._regenerate_input_feature()
        X = self.layer_info_dict[op_idx]['input_feat']  #   input after pruning of previous ops
        Y = self.layer_info_dict[op_idx]['output_feat']  # fixed output from original model
        weight = op.weight.data.cpu().numpy()
        # conv [C_out, C_in, ksize, ksize]
        # fc [C_out, C_in]
        op_type = 'Conv2D'
        if len(weight.shape) == 2:
            op_type = 'Linear'
            weight = weight[:, :, None, None]
        extract_t2 = time.time()
        self.extract_time += extract_t2 - extract_t1
        fit_t1 = time.time()

        if preserve_idx is None:  # not provided, generate new
            importance = np.abs(weight).sum((0, 2, 3))
            sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
            preserve_idx = sorted_idx[:d_prime]  # to preserve index
        assert len(preserve_idx) == d_prime
        mask = np.zeros(weight.shape[1], bool)
        mask[preserve_idx] = True

        # reconstruct, X, Y <= [N, C]
        masked_X = X[:, mask]
        if weight.shape[2] == 1:  # 1x1 conv or fc
            from lib.utils import least_square_sklearn
            rec_weight = least_square_sklearn(X=masked_X, Y=Y)
            rec_weight = rec_weight.reshape(-1, 1, 1, d_prime)  # (C_out, K_h, K_w, C_in')
            rec_weight = np.transpose(rec_weight, (0, 3, 1, 2))  # (C_out, C_in', K_h, K_w)
        else:
            assert weight.shape[2] == 3 
            rec_weight = weight[:,mask,:,:]
            # raise NotImplementedError('Current code only supports 1x1 conv now!')
        if not self.export_model:  # pad, pseudo compress
            rec_weight_pad = np.zeros_like(weight)
            rec_weight_pad[:, mask, :, :] = rec_weight
            rec_weight = rec_weight_pad

        if op_type == 'Linear':
            rec_weight = rec_weight.squeeze()
            assert len(rec_weight.shape) == 2
        fit_t2 = time.time()
        self.fit_time += fit_t2 - fit_t1
        # now assign
        op.weight.data = torch.from_numpy(rec_weight).cuda()
        action = np.sum(mask) * 1. / len(mask)  # calculate the ratio
        if self.export_model:  # prune previous buffer ops
            prev_idx = self.prunable_idx[self.prunable_idx.index(op_idx) - 1]
            for idx in range(prev_idx, op_idx):
                m = m_list[idx]
                if type(m) == nn.Conv2d:  # depthwise
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask, :, :, :]).cuda()
                    if m.groups == m.in_channels:
                        m.groups = int(np.sum(mask))
                elif type(m) == nn.BatchNorm2d:
                    m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
                    m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
                    m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
                    m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        return action, d_prime, preserve_idx


def least_square_sklearn(X, Y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    return reg.coef_
