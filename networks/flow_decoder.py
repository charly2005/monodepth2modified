from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


# First classify if a point is dynamic/static
# Then only compute the 2D dynamic flow on the dynamic regions

class FlowDecoder(nn.Module):

    def __init__(self,
                 num_ch_enc,
                 num_input_features,
                 num_frames_to_predict_for = None,
                 stride = 1
                 ):
        super(FlowDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        # num_frames_to_predict_for is output not input

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("mask", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("mask", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("mask", 2)] = nn.Conv2d(256, num_frames_to_predict_for, 1) 

        self.convs[("flow", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("flow", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("flow", 2)] = nn.Conv2d(256, 2 * num_frames_to_predict_for, 1) 

        self.net = nn.ModuleList(list(self.convs.values()))
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        flow = cat_features
        mask = cat_features
        # for mask

        for i in range(3):
            mask = self.convs[("mask"), i](mask)
            flow = self.convs(["flow"], i)(flow)
            if i != 2:
                mask = self.relu(mask)
                flow = self.relu(flow)

        mask = self.sigmoid(mask)

        mask = mask.repeat_interleave(2, dim=1)

        flow = flow * mask

        return flow, mask



