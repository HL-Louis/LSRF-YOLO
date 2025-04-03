# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS
from mmyolo.models.backbones.lsf_yolo_backbone import RSCSPLayer
from .rf_neck import RFYOLONeck
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


class FRM(BaseModule):
    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


        self.act = nn.Softmax(dim=1)


    def forward(self, x):
        prev_shape = x.shape[2:]
        xs = self.conv1(x)
        xs = F.interpolate(xs, size=prev_shape, mode='bilinear')
        # xs = self.conv2(xs)
        xs = self.act(xs)
        xs = xs * x
        x = x + xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x




class upsample_layer(nn.Module):
    def __init__(self, in_channels):
        super(upsample_layer, self).__init__()

        self.delta_gen1 = nn.Sequential(
            ConvModule(in_channels * 2, in_channels, kernel_size=1, conv_cfg=None, norm_cfg=dict(type='BN'),
                       act_cfg=None),
            ConvModule(in_channels, 2, kernel_size=3, padding=1, conv_cfg=None, norm_cfg=None, act_cfg=None),
        )

        self.delta_gen2 = nn.Sequential(
            ConvModule(in_channels * 2, in_channels, kernel_size=1, conv_cfg=None, norm_cfg=dict(type='BN'),
                       act_cfg=None),
            ConvModule(in_channels, 2, kernel_size=3, padding=1, conv_cfg=None, norm_cfg=None, act_cfg=None),
        )


    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[w / s, h / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage


@MODELS.register_module()
class LSRF_Neck(RFYOLONeck):
    def __init__(
            self,
            in_channels: Sequence[int],
            out_channels: int,
            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            num_csp_blocks: int = 3,
            freeze_all: bool = False,
            use_depthwise: bool = False,
            expand_ratio: float = 0.5,
            upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
            conv_cfg: bool = None,
            norm_cfg: ConfigType = dict(type='BN'),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.upsample_cfg = upsample_cfg
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg


        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = ConvModule(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            layer = nn.Identity()

        return layer

    def reup_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = ConvModule(
                self.in_channels[idx - 2],
                self.in_channels[idx - 1],
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

        elif idx == len(self.in_channels) - 2:
            layer = ConvModule(
                self.in_channels[idx - 2],
                self.in_channels[idx - 1],
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()
        return layer



    def build_upsample_layer(self, idx: int, *args, **kwargs) -> nn.Module:
        return upsample_layer(self.in_channels[idx - 1])



    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """

        if idx == 2:
            return RSCSPLayer(
                self.in_channels[idx - 1] * 3,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                heterogeneous_kernel=1,
                dilation=True,
                expand_ratio=self.expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)


        else:
            return nn.Sequential(

                RSCSPLayer(
                    self.in_channels[idx - 1] * 3,
                    self.in_channels[idx - 1],
                    num_blocks=self.num_csp_blocks,
                    heterogeneous_kernel=1,
                    dilation=True,
                    expand_ratio=self.expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),

                self.conv(
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if idx == 0:
            return None
        else:
            return self.conv(
                self.in_channels[idx],
                self.in_channels[idx],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        if idx == 0:
            return None
        else:
            return RSCSPLayer(
                self.in_channels[idx] * 2,
                self.in_channels[idx + 1],
                num_blocks=self.num_csp_blocks,
                heterogeneous_kernel=1,
                dilation=True,
                expand_ratio=self.expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """

        return self.conv(
            self.in_channels[idx + 1],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 1, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            bot_feat = reduce_outs[idx - 2]
            bot_feat = self.re_up_layers[idx - 1](bot_feat)
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                feat_low, feat_high)

            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low, bot_feat], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat, bot_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(0, len(self.in_channels) - 2):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(0, len(self.in_channels) - 1):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)

