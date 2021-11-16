import os

import numpy as np

import json
import urllib
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from PackPathway import PackPathway

class VideoDataset(Dataset):
    def __init__(self, root, model_name="slowfast_r50", device="cpu"):
        self.model_name = model_name
        self.root = root
        self.sub_folder_info = [(root+'/'+i, len(os.listdir(root+'/'+i))) for i in os.listdir(root)]
        self.n = 0
        self.device = device
        for i in self.sub_folder_info:
            self.n += i[1]
        self.cum_sum_arr = [i[1] for i in self.sub_folder_info]
        for i in range(1,8):
            self.cum_sum_arr[i] = self.cum_sum_arr[i]+self.cum_sum_arr[i-1]
        
        if model_name == "x3d_s":
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            frames_per_second = 30
            model_transform_params  = {
                "x3d_xs": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 4,
                    "sampling_rate": 12,
                },
                "x3d_s": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 13,
                    "sampling_rate": 6,
                },
                "x3d_m": {
                    "side_size": 256,
                    "crop_size": 256,
                    "num_frames": 16,
                    "sampling_rate": 5,
                }
            }

            # Get transform parameters based on model
            transform_params = model_transform_params[model_name]

            # Note that this transform is specific to the slow_R50 model.
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(transform_params["num_frames"]),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(
                            crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                        )
                    ]
                ),
            )

            # The duration of the input clip is also specific to the model.
            self.clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
        
        if model_name == "slowfast_r50":
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            frames_per_second = 30
            
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(
                            size=side_size
                        ),
                        CenterCropVideo(crop_size),
                        PackPathway()
                    ]
                ),
            )
            self.clip_duration = (num_frames * sampling_rate)/frames_per_second
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        lbl = 0
        for i,ele in enumerate(self.cum_sum_arr):
            if ele > index:
                lbl = i
                break
        tmp = 0
        if lbl != 0:
            tmp = self.cum_sum_arr[lbl-1]
        rmd = index-tmp
        video = EncodedVideo.from_path(self.sub_folder_info[lbl][0]+'/'+os.listdir(self.sub_folder_info[lbl][0])[rmd])
        video_data = video.get_clip(start_sec=0, end_sec=video.duration)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        inputs.to(self.device)

        # label = np.array([0,0,0,0,0,0,0,0], dtype="float32")
        # label[lbl] = 1
        # label = np.array([0])
        # label[0] = lbl

        # if self.model_name == "slowfast_r50":
        #     return torch.Tensor(tmp[0]), lbl
        # elif self.model_name == "x3d_s":
        #     return inputs[None, ...][0], lbl
        return inputs[None, ...][0], lbl
