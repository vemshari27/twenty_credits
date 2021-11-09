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

def train(opt):
    device, model_name, training_data, validation_data, output, workers, batch_size, num_epochs = opt.device, opt.model_name, opt.training_data, opt.validation_data, opt.output, opt.workers, opt.batch_size, opt.num_epochs

    # creating output folder
    if not os.path.isdir(output):
        os.mkdir(output)
    
    # # creating log file
    # f = open("training_log.txt", "w")

    # creating dataloaders
    trainset = VideoDataset(training_data, model_name, device=device)
    testset = VideoDataset(validation_data, model_name, device=device)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size, num_workers=workers)

    # loading the model
    model = VideoClassifier(model_name)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    curr_best_acc = 0

    for epoch in range(num_epochs):
        print("Epoch: ", epoch+1)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        
        # validation
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the test data: %d %%' % (
            100 * correct / total))

        torch.save(model.state_dict(), output+"/last_epoch.pt")
        if (100 * correct / total) > curr_best_acc:
            curr_best_acc = 100 * correct / total
            torch.save(model.state_dict(), output+"/best_epoch.pt")

    print('Finished Training')
    # f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--validation_data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    # parser.add_argument("--do_split", action="store_true")
    opt = parser.parse_args()

    train(opt)
