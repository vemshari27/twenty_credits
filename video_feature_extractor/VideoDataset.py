import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, video_file) -> None:
        self.video = EncodedVideo.from_path(video_file)
        self.video_duration = video.duration
    
    def __len__(self):
        return self.video_duration/6
    
    def __getitem__(self, idx):
        start_sec = idx*6
        end_sec = start_sec+3
        video_data = self.video.get_clip(start_sec=start_sec, end_sec=min(end_sec, self.video_duration))
