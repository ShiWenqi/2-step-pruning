import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
import time
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
import torchvision.transforms as transforms
import os
from main import FilterPruner


# VGG16 based network for classifying between dogs and cats.
# After training this will be an over parameterized network,
# with potential to shrink it.
def train(optimizer=None, rankfilters=False):
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if rankfilters:
            outputs = pruner.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = pruner.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()  # item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self, model=None):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        self.classifier = nn.Sequential(
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Trainer:
    def __init__(self, train_path, test_path, model, optimizer):
        self.train_data_loader = loader
        self.test_data_loader = test_loader

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

    def test(self):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0
        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            t0 = time.time()
            output = model(Variable(batch)).cpu()
            t1 = time.time()
            total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)
        print("Average prediction time", float(total_time) / (i + 1), i + 1)

        self.model.train()

    def train(self, epoches=10):
        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, batch, label):
        self.model.zero_grad()
        input = Variable(batch)
        self.criterion(self.model(input), Variable(label)).backward()
        self.optimizer.step()

    def train_epoch(self):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(batch.cuda(), label.cuda())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
                        help="Use cp decomposition. uses tucker by default")
    parser.add_argument("--test_decompose", dest="test_decompose", action="store_true")
    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)
    parser.set_defaults(store_true=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    tl.set_backend('pytorch')

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
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.train:
        model = ModifiedVGG16Model().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)
        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.train(epoches=10)
        torch.save(model, "model")

    elif args.decompose:
        model = torch.load("model").cuda()
        model.eval()
        model.cpu()
        N = len(model.features._modules.keys())
        for i, key in enumerate(model.features._modules.keys()):

            if i >= N - 2:
                break
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape) // 3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)

                model.features._modules[key] = decomposed

            torch.save(model, 'decomposed_model')

    elif args.fine_tune:
        base_model = torch.load("decomposed_model")
        model = torch.nn.DataParallel(base_model)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()

        if args.cp:
            optimizer = optim.SGD(model.parameters(), lr=0.000001)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        trainer = Trainer(args.train_path, args.test_path, model, optimizer)

        trainer.test()
        model.cuda()
        model.train()
        trainer.train(epoches=100)
        model.eval()
        trainer.test()

    elif args.test_decompose:
        criterion = nn.CrossEntropyLoss()
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.prune')
        net = checkpoint['net']
        acc = checkpoint['acc']
        pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
        total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

        use_cuda = 0
        cfg = pruner.get_cfg()
        conv_index_max = pruner.get_conv_index_max()
        last_conv_index = 0
        pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
        module_index = [key for i, key in enumerate(pruner.model.features._modules.keys())
                        if isinstance(pruner.model.features._modules[key], torch.nn.modules.conv.Conv2d)]

        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = pruner.model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data[0]  # loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        acc = 100. * correct / total

        for index in range(len(cfg)):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)

            if cfg[index] == 'M':
                conv_layer = pruner.model.features._modules[module_index[last_conv_index]]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape) // 3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)
                print(decomposed)
                pruner.model.features._modules[module_index[last_conv_index]] = decomposed

                pruner.model.cuda()
                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                    outputs = pruner.model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.data[0]  # loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc = 100. * correct / total

                optimizer = torch.optim.SGD(pruner.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
                for epoch in range(10):
                    train(optimizer)
                optimizer = torch.optim.SGD(pruner.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
                for epoch in range(5):
                    train(optimizer)

                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                    outputs = pruner.model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.data[0]  # loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc = 100. * correct / total

            if index + 1 < len(cfg):
                if not isinstance(cfg[index + 1], str):
                    last_conv_index += 1
