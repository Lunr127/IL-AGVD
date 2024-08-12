import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image

class XCLIPDataset(Dataset):
    def __init__(self, csv_file, max_frames=8):
        self.df = pd.read_csv(csv_file)
        self.index_list = self.df.index.tolist()
        self.max_frames = max_frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_frame_path = row['video_frame_path']
        frame_list = eval(row['frame_seq'])
        frames = []

        # 仅选择 max_frames 数量的帧
        if len(frame_list) > self.max_frames:
            frame_list = frame_list[:self.max_frames]
        
        for frame_num in frame_list:
            frame_path = os.path.join(video_frame_path, f"{frame_num}.jpg")
            try:
                image = Image.open(frame_path).convert('RGB')
                image = np.array(image.resize((224, 224)))  # Resize to 224x224
                frames.append(image.transpose(2, 0, 1)[np.newaxis, :])  # Convert to (C, H, W) and add batch dimension
            except Exception as e:
                print(f"Warning: {e} for file {frame_path}. Skipping frame.")
                frames.append(np.zeros((3, 224, 224))[np.newaxis, :])  # Add a blank frame if loading fails

        frames = np.concatenate(frames, axis=0)  # Combine all frames into one tensor
        frames = torch.tensor(frames, dtype=torch.float32)  # Convert to tensor

        # 如果帧数不足 max_frames，进行填充
        if frames.shape[0] < self.max_frames:
            padding = torch.zeros((self.max_frames - frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]), dtype=frames.dtype)
            frames = torch.cat([frames, padding], dim=0)

        return idx, frames

    def __len__(self):
        return len(self.df)



def create_dataloader(csv_file, batch_size, num_workers=4):
    dataset = XCLIPDataset(csv_file)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return dataloader

