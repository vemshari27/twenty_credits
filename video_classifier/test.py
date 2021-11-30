import argparse
import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from VideoClassifier import VideoClassifier
from VideoDataset import VideoDataset

def test(opt):
    device, model_name, validation_data, output, workers, batch_size = opt.device, opt.model_name, opt.validation_data, opt.output, opt.workers, opt.batch_size

    # creating output folder
    if not os.path.isdir(output):
        os.mkdir(output)
    
    # creating dataloaders
    testset = VideoDataset(validation_data, model_name, device=device)
    testloader = torch.utils.data.DataLoader(testset, batch_size, num_workers=workers)
    print("Testing", len(testset), "videos")

    # loading the model
    model = VideoClassifier(model_name)
    model.to(device)
    
    # validation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the test data: %d %%' % (
        100 * correct / total))
    print('Finished Testing')
    # f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--validation_data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--do_split", action="store_true")
    opt = parser.parse_args()

    test(opt)
