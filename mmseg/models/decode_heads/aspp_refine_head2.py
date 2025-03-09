import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import init
from mmseg.ops import resize
from ..builder import HEADS
from .refine_decode_head import RefineBaseDecodeHead
from .aspp_head import ASPPModule
from .ocr_head import SpatialGatherModule,ObjectAttentionBlock





@HEADS.register_module()
class RefineASPPHead(RefineBaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """


    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(RefineASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.attention=SpatialGatherModule(scale=1)
        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.fusion=ObjectAttentionBlock(
            self.channels,
            self.channels//2,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        fm_middle = []
        x = self.bottleneck(x)
        output = self.cls_seg(x)
        context = self.attention(x, output)
        feats = self.fusion(x, context)
        output = self.cls_seg(feats)
        fm_middle.append(feats)
        fm_middle.append(output)
        return output, fm_middle
