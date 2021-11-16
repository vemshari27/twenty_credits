import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models import x3d, slowfast

class VideoClassifier(nn.Module):
    def __init__(self, model_name="slowfast_r50"):
        super(VideoClassifier, self).__init__()
        self.model = None
        self.model_name = model_name
        if self.model_name == "x3d_s":
            self.model = x3d.create_x3d(model_num_class=8)
        if self.model_name == "slowfast_r50":
            self.model = slowfast.create_slowfast(model_num_class=8)

        
    def forward(self, X):
        # x = F.relu(self.fc1(X))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # output = F.log_softmax(x, dim=1)

        output = self.model(X)
        return output
