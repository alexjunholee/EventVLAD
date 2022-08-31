from networks.base import BaseModel
import torch.nn as nn
import torch
import numpy as np
from .unet_rpg import UNet, UNetRecurrent
from os.path import join
from .submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True

class RecurrUNet(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, num_bins = 3, in_channels=1, out_channels=2, depth=4, slope=0.2):
        self.output_channels = out_channels
        self.num_encoders = depth
        self.base_num_channels = out_channels
        self.num_residual_blocks = depth
        self.in_channels = in_channels
        self.num_bins = num_bins  # number of bins in the voxel grid event tensor
        config = {}
        super(RecurrUNet, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convgru'  # or 'convlstm'

#        self.unetrecurrent = UNet(num_input_channels=self.in_channels,
#                                           num_output_channels=self.output_channels,
#                                           skip_type='sum',
#                                           activation='sigmoid',
#                                           num_encoders=self.num_encoders,
#                                           base_num_channels=self.base_num_channels,
#                                           num_residual_blocks=self.num_residual_blocks,
#                                           norm=self.norm,
#                                           use_upsample_conv=self.use_upsample_conv)
        self.unetrecurrent = UNetRecurrent(num_input_channels=self.in_channels,
                                           num_output_channels=self.output_channels,
                                           skip_type='sum',
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
#        img_pred = self.unetrecurrent.forward(event_tensor)
        states = None
        num_bins = event_tensor.shape[1]
        for nth in range(num_bins):
            eventimg = event_tensor[:,nth,]
            eventimg = eventimg[:,np.newaxis,]
            img_pred, states = self.unetrecurrent.forward(eventimg, states)
        return img_pred
