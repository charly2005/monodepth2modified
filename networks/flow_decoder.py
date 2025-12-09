from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


# First classify if a point is dynamic/static
# Then only compute the 2D dynamic flow on the dynamic regions

class FlowDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_frames_to_predict_for = None, use_skips=True):
        super(FlowDecoder, self).__init__()

        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("maskconv",s)] = Conv3x3(self.num_ch_dec[s], self.num_frames_to_predict_for) # bin mask
            self.convs[("flowconv", s)] = Conv3x3(self.num_ch_dec[s], 2 * self.num_frames_to_predict_for)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
      
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                batch_size, _, h, w = x.shape
                
                soft_mask = self.convs[("maskconv", i)](x)
                soft_mask = self.sigmoid(soft_mask)
                hard_mask = (soft_mask > 0.5).float()
                mask = hard_mask.detach() - soft_mask.detach() + soft_mask
                mask = mask.view(batch_size, self.num_frames_to_predict_for, 1, h, w)

                flow = self.convs[("flowconv", i)](x)
                flow = flow.view(batch_size, self.num_frames_to_predict_for, 2, h, w)
                
                mask_expanded = mask.expand(-1, -1, 2, -1, -1)
                masked_flow = flow * mask_expanded
                
                # (batch_size, num_frames, 2, h, w)
                # outpust[("flow",scale)][:,i]
                self.outputs[("flow", i)] = masked_flow
                self.outputs[("mask", i)] = mask

        return self.outputs


