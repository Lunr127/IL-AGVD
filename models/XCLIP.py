from transformers import XCLIPVisionModel
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from transformers import XCLIPVisionModel

class XCLIP(nn.Module):
    def __init__(self, channel_size=512, dropout=0.2, class_num=1):
        super(XCLIP, self).__init__()
        # 加载预训练的XCLIP视觉模型
        self.backbone = XCLIPVisionModel.from_pretrained("/vhome/lixinghan/share-ckpt/DeMamba/ckpt/xclip-base-patch32")
        # 定义LayerNorm层
        self.fc_norm = nn.LayerNorm(768)

    def forward(self, x):
        b, t, _, h, w = x.shape  # 获取输入视频的批次大小、时间帧数、高度和宽度
        images = x.view(b * t, 3, h, w)  # 展平为图像列表
        outputs = self.backbone(images, output_hidden_states=True)  # 获取模型输出
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)  # 重塑为(batch_size, time_steps, hidden_size)
        video_level_features = self.fc_norm(sequence_output.mean(1))  # 计算视频级别的特征

        return video_level_features


