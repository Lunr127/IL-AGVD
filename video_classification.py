import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import create_dataloader
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import defaultdict, Counter
from transformers import XCLIPVisionModel

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

def process_csv_files_in_directory(directory, batch_size=8, num_workers=4):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    print(csv_files)
    all_video_features = {}

    for csv_file_path in csv_files:
        print(f"Processing file: {csv_file_path}")
        method_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        dataloader = create_dataloader(csv_file_path, batch_size, num_workers)
        
        model = XCLIP()
        model.eval()

        # 用于存储当前 CSV 文件的所有视频特征
        video_features = []

        with torch.no_grad():
            for idx, frames in dataloader:
                # frames = frames.cuda()  # 如果使用 GPU，确保数据加载到 GPU 上
                features = model(frames)  # 提取视频级别的特征
                video_features.append(features.cpu().numpy())

        video_features = np.concatenate(video_features, axis=0)
        
        all_video_features[method_name] = video_features

    return all_video_features

def classify_videos(all_video_features, cluster_num=5):
    method_clusters = {}

    # 将所有方法中的视频特征合并，用于训练 KMeans
    all_features = np.concatenate(list(all_video_features.values()), axis=0)
    
    # 使用 KMeans 聚类方法进行分类
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    kmeans.fit(all_features)

    for method_name, features in all_video_features.items():
        # 对每个视频特征进行分类
        predicted_labels = kmeans.predict(features)
        
        # 统计每个方法中不同类别的视频数量
        label_counts = Counter(predicted_labels)
        most_common_label = label_counts.most_common(1)[0][0]  # 获取出现最多的类别
        
        method_clusters[method_name] = {
            'predicted_cluster': most_common_label,
            'label_counts': label_counts
        }

    return method_clusters

def main(csv_directory):
    # 从 CSV 文件提取所有视频的特征
    all_video_features = process_csv_files_in_directory(csv_directory, batch_size=8, num_workers=8)
    
    # 假设初步聚类数目
    cluster_num = 5  # 初始聚类数量，可以根据实际情况调整
    
    # 对视频进行分类，并确定每个方法的类别
    method_clusters = classify_videos(all_video_features, cluster_num)
    
    # 输出聚类结果
    for method_name, cluster_info in method_clusters.items():
        print(f"Method: {method_name}, Predicted Cluster: {cluster_info['predicted_cluster']}, Label Counts: {cluster_info['label_counts']}")

    return method_clusters

if __name__ == "__main__":
    csv_directory = '/vhome/lixinghan/share-data/Gen-Video/datasets/val_csv'
    method_clusters = main(csv_directory)
