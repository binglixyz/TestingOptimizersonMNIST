# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:43:40 2017

@author: bingl
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5) #1x28x28 -> 10x24x24
        self.max_polling1 = nn.MaxPool2d(4)#10x24x24 -> 10x6x6
        self.linear1 = nn.Linear(360,10)
        
    def forward(self, data):
        conv1_out = self.conv1(data)
        max_polling1_out = self.max_polling1(conv1_out)
        flatten_out = max_polling1_out.view(-1, 360)
        linear1_out = self.linear1(flatten_out)
        return F.log_softmax(linear1_out)

root = "./data"
train_loader = torch.utils.data.DataLoader(datasets.MNIST(root, train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), download=True), batch_size= 128, shuffle= True, drop_last=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), download=True), batch_size= 128, shuffle= True, drop_last=True)

net = Net()
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.5, nesterov = False) 
#optimizer = optim.Adagrad(net.parameters(), lr=0.01)
#optimizer = optim.Adadelta(net.parameters(), lr=0.01)
#optimizer = optim.RMSprop(net.parameters(), lr=0.01)
#optimizer = optim.Adam(net.parameters())

def train(epoch):
    losses = []
    for epoch_idx in range(epoch):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            predicted = net(data)
            loss = F.nll_loss(predicted, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        average_loss = 128*total_loss/len(train_loader.dataset)
        losses = losses + [average_loss]
    for epoch_idx, epoch_loss in enumerate(losses):
        print(epoch_loss)

def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

train(10)
test()