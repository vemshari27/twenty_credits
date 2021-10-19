import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoFeatureClassifier(nn.Module):
    def __init__(self, model_name="slowfast_r50"):
        super().__init__()
        self.model = None
        self.model_name = model_name
        if self.model_name == "slowfast_r50":
            fc1 = nn.Linear(400,256)
            fc2 = nn.Linear(256,128)
            fc3 = nn.Linear(128,64)
            fc4 = nn.Linear(64,32)
            fc5 = nn.Linear(32,8)
            self.model = nn.Sequential(fc1, nn.ReLU(),
                                        fc2, nn.ReLU(),
                                        fc3, nn.ReLU(),
                                        fc4, nn.ReLU(),
                                        fc5, nn.Softmax(dim=1))
        
        elif self.model_name == "x3d_s":
            conv_layer = nn.Conv2d(2048, 1024, 2)
            fc1 = nn.Linear(1024, 256)
            fc2 = nn.Linear(256, 8)
            sl = nn.Softmax(dim=1)
            al = nn.AdaptiveAvgPool3d(output_size=1)
            self.model = nn.Sequential(fc1, fc2, sl, al)
            

    def forward(self, X):
        # x = F.relu(self.fc1(X))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # output = F.log_softmax(x, dim=1)

        output = self.model(X)
        return output