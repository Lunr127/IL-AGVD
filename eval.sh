#!/bin/bash
#SBATCH --job-name=train             # 作业名称
#SBATCH --output=/vhome/lixinghan/share-ckpt/IL-AGVD/eval_%j.out  # 标准输出和错误输出的文件名（%j 会被替换为作业ID）
#SBATCH --ntasks=1                           # 运行的任务数
#SBATCH --time=2-00:00:00                      # 预计执行时间（小时:分钟:秒）
#SBATCH --gres=gpu:0                         # 请求GPU资源
#SBATCH --mem=128G                            # 请求的系统内存
#SBATCH --partition=fvl
#SBATCH --nodes=1                            # 请求使用的节点数
#SBATCH --cpus-per-task=8                    # 每个任务请求的CPU核心数量
#SBATCH --qos=low                         # 请求高优先级的队列
#SBATCH --nodelist=fvl12              # 指定节点

nvidia-smi

# 运行你的命令
python /vhome/lixinghan/project/IL-AGVD/video_classification.py

