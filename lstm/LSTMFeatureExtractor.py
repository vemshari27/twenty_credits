from Convolutional_LSTM_PyTorch-master import ConvLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.clstm = ConvLSTM(input_channels=2048, hidden_channels=[1024], kernel_size=2)
        fc1 = nn.Linear(1024,256)
        fc2 = nn.Linear(256,128)
        fc3 = nn.Linear(128,64)
        fc4 = nn.Linear(64,32)
        fc5 = nn.Linear(32,1)
        self.model = nn.Sequential(fc1, nn.ReLU(),
                                        fc2, nn.ReLU(),
                                        fc3, nn.ReLU(),
                                        fc4, nn.ReLU(),
                                        fc5, nn.Softmax(dim=1))
    
    def forward(self, X):
        x = self.clstm(X)
        output = self.model(x)
        return output