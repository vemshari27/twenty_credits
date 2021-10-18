import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoFeatureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(400,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,8)

    def forward(self, X):
        x = F.relu(self.fc1(X))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        output = F.log_softmax(x, dim=1)
        return output