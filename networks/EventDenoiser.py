#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

import torch.nn as nn
import torch
from .DnCNN import DnCNN
from .ConvLSTM import RecurrUNet

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class EventDenoiser(nn.Module):
    def __init__(self, input_images, dep_S=5, dep_U=4, slope=0.2):
        super(EventDenoiser, self).__init__()
        config = {'num_bins' : 3}
        self.ReconNet = RecurrUNet(num_bins = 3, in_channels = 1, out_channels = 1, depth=dep_U, slope=slope)
        self.ErrorNet = DnCNN(in_channels = 3, out_channels = 1, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x):
        img_estim = self.ReconNet(x)
        err_estim = self.ErrorNet(x)
        evterr = torch.cat((img_estim,err_estim),dim=1)
        return evterr
