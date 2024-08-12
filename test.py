import models
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import XCLIPVisionModel
from dataloader import create_dataloader

class XCLIP(nn.Module):
    def __init__(self, channel_size=512, dropout=0.2, class_num=1):
        super(XCLIP, self).__init__()
        # 加载预训练的 XCLIP 视觉模型
        self.backbone = XCLIPVisionModel.from_pretrained("/vhome/lixinghan/share-ckpt/DeMamba/ckpt/xclip-base-patch32")
        # 定义 LayerNorm 层
        self.fc_norm = nn.LayerNorm(768)

    def forward(self, x):
        b, t, _, h, w = x.shape  # 获取输入视频的批次大小、时间帧数、高度和宽度
        images = x.view(b * t, 3, h, w)  # 展平为图像列表
        outputs = self.backbone(images, output_hidden_states=True)  # 获取模型输出
        sequence_output = outputs['pooler_output'].reshape(b, t, -1)  # 重塑为(batch_size, time_steps, hidden_size)
        video_level_features = self.fc_norm(sequence_output.mean(1))  # 计算视频级别的特征

        return video_level_features

# 初始化模型
model = XCLIP()

# 将模型设置为评估模式
model.eval()

# 加载数据集
# 获取目录中所有CSV文件的路径
directory = "/vhome/lixinghan/share-data/Gen-Video/datasets/train_csv"
csv_files = glob.glob(os.path.join(directory, "*.csv"))

for csv_file_path in csv_files:
    print(f"Processing file: {csv_file_path}")
    
    # 创建数据加载器
    dataloader = create_dataloader(csv_file_path, batch_size=4, num_workers=4)
    
    # 用于存储提取的特征
    all_features = []
    
    # 使用模型提取特征
    with torch.no_grad():
        for idx, frames in dataloader:
            frames = frames.cuda()  # 如果使用 GPU，确保数据加载到 GPU 上
            features = model(frames)  # 提取视频级别的特征
            all_features.append(features.cpu().numpy())  # 将特征移动到 CPU 并转换为 numpy 数组
            
    # 将提取的所有特征转换为 numpy 数组
    all_features = np.concatenate(all_features, axis=0)
    
    print(f"提取的特征形状: {all_features.shape}")

