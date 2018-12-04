import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import math
from heapq import nsmallest
from operator import itemgetter
import json
import numpy as np
import argparse
from models import *
from model_refactor import *
from modified_googlenet import *
from models import modified_resnet18
from PIL import Image
import fpzip
from numcompress import compress, decompress
from torch.autograd import Variable
from torchsummary import summary


NUM_CLASSES = 10


class BranchyAlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(BranchyAlexNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu_1 = nn.ReLU(inplace=True)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu_2 = nn.ReLU(inplace=True)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_3 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)
        self.conv_4 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.relu_4 = nn.ReLU(inplace=True)
        self.conv_5 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.relu_5 = nn.ReLU(inplace=True)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc_1 = nn.Linear(1024, 256)
        self.relu_6 = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout()
        self.fc_2 = nn.Linear(256, 128)
        self.relu_7 = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout()
        self.fc_3 = nn.Linear(128, 10)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.branch_1_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 10))

        self.branch_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.branch_2_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 10))

        self.branch_3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.branch_3_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 10))

        self.branch_4 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.branch_4_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 128),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        branch_1 = self.branch_1(x)
        branch_1 = branch_1.view(branch_1.size(0), 1024)
        branch_1 = self.branch_1_linear(branch_1)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.pool_2(x)
        branch_2 = self.branch_2(x)
        branch_2 = branch_2.view(branch_2.size(0), 1024)
        branch_2 = self.branch_2_linear(branch_2)
        x = self.conv_3(x)
        x = self.relu_3(x)
        branch_3 = self.branch_3(x)
        branch_3 = branch_3.view(branch_3.size(0), 1024)
        branch_3 = self.branch_3_linear(branch_3)
        x = self.conv_4(x)
        x = self.relu_4(x)
        branch_4 = self.branch_4(x)
        branch_4 = branch_4.view(branch_4.size(0), 1024)
        branch_4 = self.branch_4_linear(branch_4)
        x = self.conv_5(x)
        x = self.relu_5(x)
        x = self.pool_3(x)
        x = x.view(x.size(0), 1024)
        x = self.fc_1(x)
        x = self.relu_6(x)
        x = self.dropout_1(x)
        x = self.fc_2(x)
        x = self.relu_7(x)
        x = self.dropout_2(x)
        x = self.fc_3(x)
        return branch_1, branch_2, branch_3, branch_4, x

    def forward_track(self):
        self.layer_index = {}
        index = 0
        for layer, (name, module) in enumerate(self._modules.items()):
            if isinstance(module, torch.nn.modules.ReLU) \
                    or isinstance(module, torch.nn.modules.Conv2d) \
                    or isinstance(module, torch.nn.modules.MaxPool2d) \
                    or isinstance(module, torch.nn.modules.LocalResponseNorm) \
                    or isinstance(module, torch.nn.modules.Dropout) \
                    or isinstance(module, torch.nn.modules.Linear):
                self.layer_index[index] = layer
                index += 1
        return self.layer_index


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 BranchyNet')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if os.name == 'nt':  # windows
        num_workers = 0
    else:  # linux
        num_workers = 8
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    use_cuda = torch.cuda.is_available()
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch
    total_filter_num_pre_prune = 0
    batch_size = 32

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if args.train:
        net = BranchyAlexNet()
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        for i in range(80):
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs_branch_1, outputs_branch_2, outputs_branch_3, outputs_branch_4, outputs = net(inputs)
                loss = criterion(outputs_branch_1, targets) + criterion(outputs_branch_2, targets) + \
                       criterion(outputs_branch_3, targets) + criterion(outputs_branch_4, targets) + criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.data[0]  # item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        for i in range(40):
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs_branch_1, outputs_branch_2, outputs_branch_3, outputs_branch_4, outputs = net(inputs)
                loss = criterion(outputs_branch_1, targets) + criterion(outputs_branch_2, targets) + \
                       criterion(outputs_branch_3, targets) + criterion(outputs_branch_4, targets) + criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.data[0]  # item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

        torch.save(net, './checkpoint/ckpt.train.alexnet')
    
    if args.test:
        net = torch.load('./checkpoint/ckpt.train.alexnet')
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_loss_branch_1 = 0
        test_loss_branch_2 = 0
        test_loss_branch_3 = 0
        test_loss_branch_4 = 0
        correct = 0
        correct_branch_1 = 0
        correct_branch_2 = 0
        correct_branch_3 = 0
        correct_branch_4 = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs_branch_1, outputs_branch_2, outputs_branch_3, outputs_branch_4, outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss_branch_1 = criterion(outputs_branch_1, targets)
            loss_branch_2 = criterion(outputs_branch_2, targets)
            loss_branch_3 = criterion(outputs_branch_3, targets)
            loss_branch_4 = criterion(outputs_branch_4, targets)

            test_loss += loss.item()
            test_loss_branch_1 += loss_branch_1.item()
            test_loss_branch_2 += loss_branch_2.item()
            test_loss_branch_3 += loss_branch_3.item()
            test_loss_branch_4 += loss_branch_4.item()

            _, predicted = torch.max(outputs.data, 1)
            _, predicted_branch_1 = torch.max(outputs_branch_1.data, 1)
            _, predicted_branch_2 = torch.max(outputs_branch_2.data, 1)
            _, predicted_branch_3 = torch.max(outputs_branch_3.data, 1)
            _, predicted_branch_4 = torch.max(outputs_branch_4.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_branch_1 += predicted_branch_1.eq(targets.data).cpu().sum()
            correct_branch_2 += predicted_branch_2.eq(targets.data).cpu().sum()
            correct_branch_3 += predicted_branch_3.eq(targets.data).cpu().sum()
            correct_branch_4 += predicted_branch_4.eq(targets.data).cpu().sum()

        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss_branch_1 / (batch_idx + 1), 100. * float(correct_branch_1) / total, correct_branch_1, total))
        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss_branch_2 / (batch_idx + 1), 100. * float(correct_branch_2) / total, correct_branch_2, total))
        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss_branch_3 / (batch_idx + 1), 100. * float(correct_branch_3) / total, correct_branch_3, total))
        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss_branch_4 / (batch_idx + 1), 100. * float(correct_branch_4) / total, correct_branch_4, total))

        net.cuda()
        summary(net, (3, 32, 32))
        net.cpu()
        layer_index = net.forward_track()
        print(layer_index)

        (inputs, targets) = list(testloader)[0]
        net.cpu()
        with torch.autograd.profiler.profile() as prof:
             net(Variable(inputs))
        print(prof.function_events)