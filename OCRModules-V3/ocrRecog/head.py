# -*- coding: utf-8 -*-
# @Time : 2023/3/13 9:36
# @Author : fangxuwei
# @Github Name : BADBADBADBOY
# @File : ctc_head
# @Project : SVTR

import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels,out_channels)
        else:
            self.fc1 = nn.Linear(in_channels,mid_channels)
            self.fc2 = nn.Linear(mid_channels,out_channels)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result