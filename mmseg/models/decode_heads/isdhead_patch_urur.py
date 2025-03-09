import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..builder import build_loss
from mmseg.ops import resize
from ..losses import accuracy
from .stdc_head import ShallowNet
import numpy as np
from .dysample import DySample


sigmoid_function=nn.Sigmoid()



def calc_entropy(input_tensor):
    # print(input_tensor.shape)
    lsm = nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    # print(probs.shape)
    # print(probs[0,:,0,0])
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(1)
    return entropy.detach().clone()




class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x


class ScoreHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(ScoreHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(2, 1, kernel_size=1, bias=True)

    def forward(self, x0, x1, patch_num):
        h = x1.shape[2]
        w = x1.shape[3]
        x1 = F.interpolate(x1, (x0.shape[2], x0.shape[3]), mode='bilinear', align_corners=False)
        x = torch.abs(x0 - self.conv_bn_relu(x1))

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv_out(x)

        original_x = x
        x = F.interpolate(x, (patch_num, patch_num), mode='bilinear', align_corners=False)
        x = sigmoid_function(x)
        x = x.reshape(x.shape[0], patch_num * patch_num)

        return x, original_x

class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):

        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x


def simple_fuse2(feat, feat16, B, M, ids,values,entropy,patch_num):
    size1 = int(feat16.shape[2])
    size2 = int(feat16.shape[3])
    entropy=F.interpolate(entropy, (feat16.shape[2] * patch_num, feat16.shape[3] * patch_num), mode='bilinear', align_corners=False)
    entropy = entropy / 2.08
    values=F.interpolate(values, (feat16.shape[2] * patch_num, feat16.shape[3] * patch_num), mode='bilinear', align_corners=False)
    values=sigmoid_function(values)
    values=(values+entropy)*0.5 
    loss_weights = 1 + values.detach().clone()
    # loss_weights = loss_weights * loss_weights * loss_weights
    feat = F.interpolate(feat, (feat16.shape[2] * patch_num, feat16.shape[3] * patch_num), mode='bilinear', align_corners=False)
    feat16 = feat16.view(B, M, *feat16.shape[1:])
    patches = feat.unfold(
        2, size1, size2
    ).unfold(
        3, size1, size2
    )
    values = values.unfold(
        2, size1, size2
    ).unfold(
        3, size1, size2
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(patches.shape[0], -1, *patches.shape[3:])
    values = values.permute(0, 2, 3, 1, 4, 5)
    values = values.reshape(values.shape[0], -1, *values.shape[3:])
    temp_list = []
    for number in range(patches.shape[0]):
        patches[number, ids[number]] = (1-values[number,ids[number]]) * patches[number, ids[number]] + values[number,ids[number]]*feat16[number] 
    patches = patches.permute(0, 2, 3, 4, 1)  # 2,2,25,25,100
    patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])  # 2,2*25*25,100
    fold = nn.Fold(output_size=(size1 * patch_num, size2 * patch_num), kernel_size=size1, stride=size2)
    patches = fold(patches)
    return patches,loss_weights


@HEADS.register_module()
class ISDHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels, reduce=False, **kwargs):
        super(ISDHead, self).__init__(**kwargs)
        self.down_ratio = down_ratio
        final_channel=64
        # shallow branch
        self.stdc_net = ShallowNet(in_channels=3, pretrain_model="STDCNet813M_73.91.tar")
        channels_8=256
        channels_16 = 512
        self.score_head=ScoreHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                  self.channels//2, 1, kernel_size=1)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, final_channel,
                                                final_channel // 2, self.num_classes, kernel_size=1)


        self.reduce1 = Reducer(in_channels=channels_16,reduce=final_channel, bn_relu=True)
        self.reduce2 = Reducer(in_channels=channels_8,reduce=final_channel, bn_relu=True)
        self.reduce3 = Reducer(in_channels=self.channels, reduce=final_channel, bn_relu=True)
        # self.patch_conv=ResBlock()

    def forward(self, inputs, prev_output, train_flag=True):
        """Forward function."""
        if train_flag:
            patch_num=10
            select_num=100
        else:
            patch_num=20
            select_num=200
        deep_output = prev_output[1]
        entropy = calc_entropy(deep_output)
        entropy = entropy.unsqueeze(1)
        entropy_score = F.interpolate(entropy, (patch_num, patch_num), mode='bilinear')
        entropy_score = entropy_score / 2.08
        entropy_score = entropy_score.reshape(entropy_score.shape[0], patch_num * patch_num)
        
        deep_feat = prev_output[0]
        # socres, original_scores = self.score_head(deep_feat,patch_num)
        socres, original_scores = self.score_head(prev_output[2], prev_output[3], patch_num)
        values = original_scores
        _, ids = torch.topk(socres+entropy_score,select_num,1) 

        patch_size = int(inputs.shape[2] / patch_num)
        patches = inputs.unfold(
            2, patch_size, patch_size
        ).unfold(
            3, patch_size, patch_size
        ).permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(patches.shape[0], -1, *patches.shape[3:])
        
        temp_list=[]
        for number in range(patches.shape[0]):
            temp_list.append(patches[number,ids[number]])
        patches=torch.stack(temp_list)

        
        patch_shape = patches.shape
        B, M = patch_shape[:2]
        local_shallow_feat8, local_shallow_feat16 = self.stdc_net(patches.reshape(-1, *patch_shape[2:]),level=3)

        local_shallow_feat16=self.reduce1(local_shallow_feat16)
        local_shallow_feat8=self.reduce2(local_shallow_feat8)
        deep_feat = self.reduce3(deep_feat)
        local_shallow_feat16 = F.interpolate(local_shallow_feat16,(local_shallow_feat8.shape[2], local_shallow_feat8.shape[3]),mode='bilinear', align_corners=False)
        local_shallow_feat16 = local_shallow_feat16 + local_shallow_feat8
        local_shallow_feat16,lossweights = simple_fuse2(deep_feat, local_shallow_feat16, B, M, ids,values, entropy,patch_num)
        output = self.conv_seg_aux_16(local_shallow_feat16)
        if train_flag:
            return output, lossweights
        else:
            return output


    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label,loss_weights):
        """Compute segmentation loss."""
        loss = dict()
        # print(seg_logit.shape)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss_weights = resize(
            input=loss_weights,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss_weights=loss_weights.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=loss_weights,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, loss_weights = self.forward(inputs, prev_output)
        losses = self.losses(seg_logits, gt_semantic_seg, loss_weights)
        return losses


    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output, False)
