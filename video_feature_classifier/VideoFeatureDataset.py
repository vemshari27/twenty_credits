import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class VideoFeatureDataset(Dataset):
    def __init__(self, root, model_name="slowfast_r50"):
        self.model_name = model_name
        self.root = root
        self.sub_folder_info = [(root+'/'+i, len(os.listdir(root+'/'+i))) for i in os.listdir(root)]
        self.n = 0
        for i in self.sub_folder_info:
            self.n += i[1]
        self.cum_sum_arr = [i[1] for i in self.sub_folder_info]
        for i in range(1,8):
            self.cum_sum_arr[i] = self.cum_sum_arr[i]+self.cum_sum_arr[i-1]
    
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
        tmp = np.load(self.sub_folder_info[lbl][0]+'/'+os.listdir(self.sub_folder_info[lbl][0])[rmd])
        # label = np.array([0,0,0,0,0,0,0,0], dtype="float32")
        # label[lbl] = 1
        # label = np.array([0])
        # label[0] = lbl

        if self.model_name == "slowfast_r50":
            return torch.Tensor(tmp[0]), lbl
        elif self.model_name == "x3d_s":
            return torch.Tensor(tmp[0,:,0,:,:]), lbl
