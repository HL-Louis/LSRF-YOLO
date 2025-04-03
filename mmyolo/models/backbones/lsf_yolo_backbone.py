# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Union
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import SPPFBottleneck
from .base_backbone import BaseBackbone
from ..utils import make_divisible, make_round
from mmengine.model import BaseModule
from torch import Tensor
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class SPPELAN(nn.Module):
    # spp-elan
    def __init__(
        self, c1, c2, c3
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        # self.c = c3
        self.c= int(c1 * c3)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4 * self.c, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))

@MODELS.register_module()
class LSRF_YOLO_Backbone(BaseBackbone):
    # From left to right:
    # in_channels, out_channels, num_blocks, heterogeneous_kernel, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, 0,True, False], [128, 256, 6, 1, True, False],
               [256, 512, 6, 2, True, False], [512, 1024, 3, 3, False, True]],
    }

    def __init__(
            self,
            arch: str = 'P5',
            deepen_factor: float = 1.0,
            widen_factor: float = 1.0,
            input_channels: int = 3,
            out_indices: Sequence[int] = (2, 3, 4),
            frozen_stages: int = -1,
            plugins: Union[dict, List[dict]] = None,
            use_depthwise: bool = False,
            expand_ratio: float = 0.5,
            arch_ovewrite: dict = None,
            channel_attention: bool = True,
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN'),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            norm_eval: bool = False,
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        self.use_depthwise = use_depthwise
        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg

        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels,
            out_indices,
            frozen_stages=frozen_stages,
            plugins=plugins,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:  # YOLOV8 stem
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, heterogeneous_kernel, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)

        if heterogeneous_kernel == 0:

            csp_layer = RSCSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                kernel_size=3,
                use_depthwise=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        elif heterogeneous_kernel == 1:
            csp_layer = RSCSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                kernel_size=5,
                use_depthwise=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif heterogeneous_kernel == 2:
            csp_layer = RSCSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                kernel_size=7,
                use_depthwise=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            csp_layer = RSCSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                kernel_size=9,
                use_depthwise=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        # if use_spp:
        #     spp = SPPELAN(
        #         out_channels,
        #         out_channels,
        #         1,
        #         )
        #     stage.append(spp)
        return stage


class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

        # self.act = nn.Hardsigmoid()

        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class RSCSPLayer(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 heterogeneous_kernel: Sequence[int] = (0, 1, 2, 3),
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 drop_path_rate=0.2,
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)

        # self.dilation = dilation
        self.drop_path_rate = drop_path_rate
        self.kernel_size=kernel_size
        self.heterogeneous_kernel = heterogeneous_kernel
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            RSBottleneck(
                mid_channels,
                mid_channels,
                1.0,
                self.kernel_size,
                add_identity,
                use_depthwise,

                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                # drop_path_rate=self.drop_path_rate * _ / num_blocks,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

        self.attn = ChannelAttention(2 * mid_channels)


    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        x_final = self.attn(x_final)
        return self.final_conv(x_final)






class SAdapter1(BaseModule):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.SiLU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)


        self.act = act_layer()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        x = x + xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x


class SAdapter2(BaseModule):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.SiLU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        self.act = act_layer()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)

        x = x + xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x

#

class RSBottleneck(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 kernel_size=3,
                 add_identity=True,
                 use_depthwise=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 # drop_path_rate=0.,
                 init_cfg=None):
        super(RSBottleneck, self).__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.kernel_size = kernel_size

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            3,
            1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )


        self.conv2 = conv(
            hidden_channels,
            out_channels,
            self.kernel_size,
            stride=1,
            padding=((kernel_size-1))//2,
            # dilation=4,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv3 = conv(
            hidden_channels,
            out_channels,
            self.kernel_size ,
            stride=1,
            padding=((kernel_size-1))//2,
            # dilation=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )



        self.add_identity = \
            add_identity and in_channels == out_channels
        self.edge_weights = nn.Parameter(torch.ones(3), requires_grad=True)
        self.attn1 = SAdapter2(out_channels)
        self.attn2 = SAdapter2(out_channels)
        # self.drop_path_rate = drop_path_rate

    def forward(self, x):
        dtype = x[0].dtype
        batch, _, height, width = x.size()
        identity = x
        con1 = self.conv1(x)
        y = con1
        con1 = self.conv2(con1)
        con2 = self.conv3(con1)

        con1 = rearrange(con1, 'b c h w -> b c (h w)')
        con1 = rearrange(con1, 'b c n -> b n c')
        con1 = self.attn1(con1)
        con1 = rearrange(con1, 'b n c -> b c n')
        con1 = rearrange(con1, 'b c (h w) -> b c h w', h=height)
        #
        con2 = rearrange(con2, 'b c h w -> b c (h w)')
        con2 = rearrange(con2, 'b c n -> b n c')
        con2 = self.attn2(con2)
        con2 = rearrange(con2, 'b n c -> b c n')
        con2 = rearrange(con2, 'b c (h w) -> b c h w', h=height)

        edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
        weights_sum = torch.sum(edge_weights)
        out = torch.stack(
            [(con1 * edge_weights[0] + con2 * edge_weights[1] + y * edge_weights[2]) / (weights_sum + 0.0001)], dim=-1)
        out = torch.sum(out, dim=-1)

        # return out + identity

        # if self.drop_path_rate > 0:
        #     out = drop_path(out, self.drop_path_rate, self.training)
        #     out += identity
        # return out
        if self.add_identity:
            return out + identity
        else:
            return out





