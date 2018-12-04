'''
Train & Pruning with PyTorch by hou-yz.
'''

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


# Training
def train(optimizer=None, rankfilters=False):
    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if rankfilters:
            outputs = pruner.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.data[0]  # item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))


# test
def test(log_index=-1, prune_iteration=-1):
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    if log_index == -1 or use_cuda:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]  # loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Test  Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        acc = 100. * float(correct) / total

    if log_index != -1:
        (inputs, targets) = list(testloader)[0]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # get profile
        with torch.autograd.profiler.profile() as prof:
            net(Variable(inputs))
            # print(next(net.parameters()).is_cuda)
        pruner.forward_n_track(Variable(inputs), log_index)
        cfg = pruner.get_cfg()

        # get log for time/bandwidth
        delta_ts = []
        bandwidths = []

        for i in range(len(cfg)):
            delta_ts.append(
                sum(item.cpu_time for item in prof.function_events[:pruner.conv_n_pool_to_layer[i]]) /
                np.power(10, 6) / batch_size)
            if isinstance(cfg[i], int):
                bandwidths.append(
                    int(cfg[i] * (inputs.shape[2] * inputs.shape[3]) / np.power(4, cfg[:i + 1].count('M'))))
            elif cfg[i] == 'M':
                bandwidths.append(
                    int(cfg[i - 1] * (inputs.shape[2] * inputs.shape[3]) / np.power(4, cfg[:i + 1].count('M'))))
            elif cfg[i] == 'L':
                bandwidths.append(int(1))


        data = {
            'acc': acc if use_cuda else -1,
            'index': log_index,
            'delta_t_prof': delta_ts[log_index],
            'delta_ts': delta_ts,
            'bandwidth': bandwidths[log_index],
            'bandwidths': bandwidths,
            'layer_cfg': cfg[log_index],
            'config': cfg,
            'prune_iteration': prune_iteration
        }
        return data

    return acc


# save
def save(acc, conv_index=-1, epoch=-1, prune_iteration=-1):
    print('Saving..')
    print("conv_index: ", conv_index)
    print("epoch: ", epoch)
    print("Prune_iteration: ", prune_iteration)
    try:
        # save the cpu model
        model = net.module if isinstance(net, torch.nn.DataParallel) else net
        state = {
            'net': model.cpu() if use_cuda else model,
            'acc': acc,
            'conv_index': conv_index,
            'epoch': epoch,
        }
    except:
        pass
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if args.prune and prune_iteration != -1:
        torch.save(state, './checkpoint/ckpt.prune_%d' %(prune_iteration))
    elif args.prune:
        torch.save(state, './checkpoint/ckpt.prune')
    elif prune_iteration != -1:
        torch.save(state, './checkpoint/ckpt.prune_layer_%d_iteration_%d' % (conv_index, prune_iteration))
    elif args.prune_layer and conv_index != -1:
        torch.save(state, './checkpoint/ckpt.prune_layer_%d' % conv_index)
    elif epoch != -1:
        torch.save(state, './checkpoint/ckpt.train.epoch_' + str(epoch))
    else:
        torch.save(state, './checkpoint/ckpt.train')


    # restore the cuda or cpu model
    if use_cuda:
        net.cuda()


class FilterPruner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    # forward method that gives "compute_rank" a hook
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        conv_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[conv_index] = layer
                conv_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    # forward method that tracks computation info
    def forward_n_track(self, x, log_index=-1):
        self.conv_n_pool_to_layer = {}

        index = 0
        delta_t_computations = 0
        all_conv_computations = 0  # num of conv computations to the given layer
        t0 = time.time()
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.ReLU) or isinstance(module, torch.nn.modules.MaxPool2d) or isinstance(module, torch.nn.modules.AvgPool2d):
                all_conv_computations += np.prod(x.data.shape[1:])
                self.conv_n_pool_to_layer[index] = layer
                if log_index == index:
                    delta_t = time.time() - t0
                    delta_t_computations = all_conv_computations
                    bandwidth = np.prod(x.data.shape[1:])
                index += 1

        return delta_t, delta_t_computations, bandwidth, all_conv_computations

    # for all the conv layers
    def get_conv_index_max(self):
        conv_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                conv_index += 1
        return conv_index

    # for all the relu layers and pool2d layers
    def get_cfg(self):
        cfg = []
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                cfg.append(module.out_channels)
            elif isinstance(module, torch.nn.modules.MaxPool2d):
                cfg.append('M')
            elif isinstance(module, torch.nn.modules.AvgPool2d):
                cfg.append('L')
        return cfg

    def compute_rank(self, grad):
        conv_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[conv_index]
        values = torch.sum((activation * grad), dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[
                 0, :, 0, 0].data  # compute the total 1st order taylor for each filters in a given layer

        # Normalize the rank by the filter dimensions
        values = values / (activation.size(0) * activation.size(2) * activation.size(3))

        if conv_index not in self.filter_ranks:  # set self.filter_ranks[conv_index]
            self.filter_ranks[conv_index] = torch.FloatTensor(activation.size(1)).zero_()
            if use_cuda:
                self.filter_ranks[conv_index] = self.filter_ranks[conv_index].cuda()

        self.filter_ranks[conv_index] += values
        self.grad_index += 1

    def lowest_ranking_filters(self, num, conv_index):
        data = []
        if conv_index == -1:
            for i in sorted(self.filter_ranks.keys()):
                for j in range(self.filter_ranks[i].size(0)):
                    data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
        else:
            for j in range(self.filter_ranks[conv_index].size(0)):
                data.append((self.activation_to_layer[conv_index], j, self.filter_ranks[conv_index][j]))
        return nsmallest(num, data, itemgetter(2))  # find the minimum of data[_][2], aka, self.filter_ranks[i][j]

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v)).cuda()
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune, conv_index):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune, conv_index)

        # After each of the k filters are pruned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

    def get_candidates_to_prune(self, num_filters_to_prune, conv_index):
        self.reset()
        train(rankfilters=True)
        self.normalize_ranks_per_layer()

        return self.get_pruning_plan(num_filters_to_prune, conv_index)

    def total_num_filters(self, conv_index):
        filters = 0
        i = 0
        for name, module in list(self.model.features._modules.items()):
            if isinstance(module, torch.nn.modules.Conv2d):
                if conv_index == -1:
                    filters = filters + module.out_channels
                elif conv_index == i:
                    filters = filters + module.out_channels
                i = i + 1

        return filters

    def prune(self, conv_index=-1, test_accuracy=False, prune_layer=False):
        # Get the accuracy before pruning
        prune_iteration = 0
        acc_pre_prune = test()
        not_stop = 1

        if test_accuracy:
            save(acc_pre_prune, conv_index, -1, prune_iteration)
        acc = acc_pre_prune

        # train(rankfilters=True)

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = pruner.total_num_filters(conv_index)

        if prune_layer:
            a = np.logspace(np.log10(number_of_filters), 0, 16)
            for i in range(len(a)):
                a[i] = round(a[i])
            b = []
            for i in range(len(a)):
                if a[i] not in b:
                    b.append(a[i])
            num_filters_to_prune_per_iteration = int(b[prune_iteration] - b[prune_iteration+1])
            print("b: ")
            print(b)
            print("prune iteration: ")
            print(prune_iteration)
            print("filter prune: ", num_filters_to_prune_per_iteration)
        else:
            num_filters_to_prune_per_iteration = math.ceil(number_of_filters / 16)
        while (not_stop or test_accuracy) and pruner.total_num_filters(conv_index) > num_filters_to_prune_per_iteration:
            prune_iteration += 1
            # print("Ranking filters.. ")

            prune_targets = pruner.get_candidates_to_prune(num_filters_to_prune_per_iteration, conv_index)
            num_layers_pruned = {}  # filters to be pruned in each layer
            for layer_index, filter_index in prune_targets:
                if layer_index not in num_layers_pruned:
                    num_layers_pruned[layer_index] = 0
                num_layers_pruned[layer_index] = num_layers_pruned[layer_index] + 1

            print("Layers that will be pruned", num_layers_pruned)
            print("..............Pruning filters............. ")
            if use_cuda:
                self.model.cpu()

            for layer_index, filter_index in prune_targets:
                prune_conv_layer(self.model, layer_index, filter_index)

            if use_cuda:
                self.model.cuda()
                # self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True

            print("%d / %d Filters remain." % (pruner.total_num_filters(conv_index), number_of_filters))
            # test()
            print("Fine tuning to recover from pruning iteration.")

            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
            for epoch in range(10):
                train(optimizer)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
            for epoch in range(5):
                train(optimizer)

            acc = test()
            pass

            if test_accuracy:
                save(acc, conv_index, -1, prune_iteration)

            print("acc: ", acc)
            print("acc pre prune: ", acc_pre_prune)
            if acc <= acc_pre_prune - 1.8 and not test_accuracy:
                not_stop = 0
                print("acc")
            if prune_layer and prune_iteration >= len(b) - 1:
                not_stop = 0
                print("iteration")
            if prune_layer and pruner.total_num_filters(conv_index) / number_of_filters < 0.001:
                not_stop = 0
                print("filters")

            if not_stop and not args.prune:
                num_filters_to_prune_per_iteration = int(b[prune_iteration] - b[prune_iteration + 1])
                print("filter prune: ", num_filters_to_prune_per_iteration)
                print("prune iteration: ", prune_iteration)
                print("b: ", b)

        print("Finished. Going to fine tune the model a bit more")
        for epoch in range(2):
            train(optimizer)
        test()
        pass


class VGG_encode(nn.Module):
    # QF = 100 is for png encoding
    def __init__(self, vgg_model, partition_index, QF):
        super(VGG_encode, self).__init__()
        self.vgg = vgg_model
        self.partition_index = partition_index
        self.QF = QF
        index = 0
        for layer, (name, module) in enumerate(self.vgg.features._modules.items()):
            if index <= partition_index:
                if isinstance(module, torch.nn.modules.Conv2d) or isinstance(module, torch.nn.modules.BatchNorm2d):
                    module._parameters.requires_grad = False
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
            if isinstance(module, torch.nn.modules.ReLU) or isinstance(module, torch.nn.modules.MaxPool2d) or isinstance(module, torch.nn.modules.AvgPool2d):
                index += 1

    def forward(self, x):
        index = 0
        for layer, (name, module) in enumerate(self.vgg.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.ReLU) or isinstance(module, torch.nn.modules.MaxPool2d) or isinstance(module, torch.nn.modules.AvgPool2d):
                if index == self.partition_index:
                    features = x
                    if self.QF != 100:
                        features = features.view(features.size(0) * features.size(2), -1)
                        features = features.data.numpy()
                        features = np.round(features*255)
                        im = Image.fromarray(features)
                        if im.mode != 'L':
                            im = im.convert('L')
                        im.save("temp.jpeg", quality=self.QF)
                        im_decode = Image.open("temp.jpeg")
                        encoded_data_size = os.path.getsize("temp.jpeg")
                        raw_data_size = x.size(0) * x.size(1) * x.size(2) * x.size(3) * 4
                        decode_array = np.array(im_decode)
                        decode_array = decode_array / 255
                        decode_va = Variable(torch.from_numpy(decode_array))
                        decode_va = decode_va.view(x.size()).float()

                    else:
                        features = features.view(features.size(0)*features.size(1)*features.size(2)*features.size(3), -1)
                        features = torch.squeeze(features)
                        features = features.data.numpy()
                        features = features.tolist()
                        raw_data_size = sys.getsizeof(features)
                        compressed = compress(features, precision=4)
                        encoded_data_size = sys.getsizeof(compressed)
                        decode_va = np.array(decompress(compressed))
                        decode_va = Variable(torch.from_numpy(decode_va))
                        decode_va = decode_va.view(x.size()).float()
                    # print("x:")
                    # print(x.data.numpy())
                    # print("decode:")
                    # print(decode_va.data.numpy())
                    x.data = decode_va.data
                index += 1

        out = x.view(x.size(0), -1)
        out = self.vgg.classifier(out)
        return out, raw_data_size, encoded_data_size


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--prune_layer", dest="prune_layer", action="store_true")
    parser.add_argument("--test_pruned", dest="test_pruned", action="store_true")
    parser.add_argument("--prune_layer_test_accuracy", dest="prune_layer_test_accuracy", action="store_true")
    parser.add_argument("--test_encode", dest="test_encode", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # Model
    if args.train:
        print('==> Building model..')
        net = VGG('VGG16')
        # net = VGG('VGG11')
        # net = ModifiedResNet18Model()
        # net = PreActResNet18()
        # net = ModifiedGoogleNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
    else:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.train')
        net = checkpoint['net']
        acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    if use_cuda:
        net.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        # cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
    total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)

    if args.prune:
        pruner.prune(-1, False)
        acc = test()
        save(acc)
        pass
    elif args.prune_layer:
        # this is after --prune the whole model
        conv_index_max = pruner.get_conv_index_max()
        for conv_index in range(conv_index_max):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)
            # prune given layer
            pruner.prune(conv_index, test_accuracy=False, prune_layer=True)
            acc = test()
            save(acc, conv_index)
            pass
    elif args.train or args.resume:
        for epoch in range(start_epoch, start_epoch + args.epoch):
            print('\nEpoch: %d' % epoch)
            train()
            acc = test()
            if epoch % 10 == 0:
                save(acc, -1, epoch)
                pass
        save(acc)

    elif args.prune_layer_test_accuracy:
        conv_index_max = pruner.get_conv_index_max()
        Test_accuracy = True
        for conv_index in range(conv_index_max):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            # Can using ckpt.train get better result?
            checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)
            # prune given layer
            pruner.prune(conv_index, Test_accuracy, True)
            pass

    elif args.test_encode:

        use_cuda = 0
        cfg = pruner.get_cfg()
        conv_index_max = pruner.get_conv_index_max()
        PNG_data = []

        test_index = [2, 5, 9, 13, 17]

        last_conv_index = 0  # log for checkpoint restoring, nearest conv layer
        for index in range(len(cfg)):
            if index in test_index:
                # prune_layer_test_accuracy
                for prune_iteration in range(0, 16):
                    print('===> Resuming from checkpoint..')
                    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                    if os.path.exists(
                            './checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration)):
                        checkpoint = torch.load(
                            './checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration))
                        print(
                            './checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration))
                        # checkpoint = torch.load('./checkpoint/ckpt.prune')
                        net = checkpoint['net']
                        acc = checkpoint['acc']
                        if use_cuda:
                            net.cuda()
                        net_encode = VGG_encode(net, index, 100)

                        test_loss = 0
                        total = 0
                        correct = 0
                        raw_data_size = 0
                        encoded_data_size = 0
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            if use_cuda:
                                inputs, targets = inputs.cuda(), targets.cuda()
                            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                            outputs, raw_data_size_batch, encoded_data_size_batch = net_encode(inputs)
                            raw_data_size += raw_data_size_batch
                            encoded_data_size += encoded_data_size_batch
                            loss = criterion(outputs, targets)
                            test_loss += loss.data[0]  # loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            total += targets.size(0)
                            correct += predicted.eq(targets.data).cpu().sum()

                        print('Test Loss (PNG): %.3f | Acc: %.3f%% (%d/%d)' % (
                            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                        acc = 100. * correct / total
                        print('Encode ratio (PNG): %.3f | partition index: %d | QF: %d' % (
                        encoded_data_size / raw_data_size, index, 100))
                        ratio = encoded_data_size / raw_data_size

                        data = {
                            'acc': acc,
                            'index': index,
                            'layer_cfg': cfg[index],
                            'config': cfg,
                            'prune_iteration': prune_iteration,
                            'ratio': ratio
                        }
                        PNG_data.append(data)

            if index + 1 < len(cfg):
                if not isinstance(cfg[index + 1], str):
                    last_conv_index += 1

        with open('./PNG_encode.json', 'w') as fp:
            json.dump(PNG_data, fp, indent=2)


        use_cuda = 0
        cfg = pruner.get_cfg()
        conv_index_max = pruner.get_conv_index_max()
        data = []
        last_conv_index = 0
        test_index = [2, 5, 9, 13, 17]
        potential_QF = [5, 10, 20, 30, 40, 50, 65, 80, 95]

        for (_, index) in enumerate(test_index):
            for (QF_index, QF) in enumerate(potential_QF):
                print('==> Resuming from checkpoint..')
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                checkpoint = torch.load('./checkpoint/ckpt.prune')
                net = checkpoint['net']
                original_acc = checkpoint['acc']
                net_encode = VGG_encode(net, index, QF)

                test_loss = 0
                total = 0
                correct = 0
                raw_data_size = 0
                encoded_data_size = 0
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                    outputs, raw_data_size_batch, encoded_data_size_batch = net_encode(inputs)
                    raw_data_size += raw_data_size_batch
                    encoded_data_size += encoded_data_size_batch
                    loss = criterion(outputs, targets)
                    test_loss += loss.data[0]  # loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                print('Test Loss before fine tune: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc = 100. * correct / total
                print('Encode ratio: %.3f | partition index: %d | QF: %d' % (encoded_data_size / raw_data_size, index, QF))
                ratio = encoded_data_size / raw_data_size

                params = filter(lambda p: p.requires_grad, net_encode.parameters())
                optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
                print("Fine tuning to recover from JPEG encoding.")
                for epoch in range(2):
                    train_loss = 0
                    correct = 0
                    total = 0
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        if use_cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()
                        optimizer.zero_grad()
                        inputs, targets = Variable(inputs), Variable(targets)
                        outputs, raw_data_size_batch, encoded_data_size_batch = net_encode(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.data[0]  # item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.data).cpu().sum()
                    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                test_loss = 0
                total = 0
                correct = 0
                raw_data_size = 0
                encoded_data_size = 0
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                    outputs, raw_data_size_batch, encoded_data_size_batch = net_encode(inputs)
                    raw_data_size += raw_data_size_batch
                    encoded_data_size += encoded_data_size_batch
                    loss = criterion(outputs, targets)
                    test_loss += loss.data[0]  # loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                print('Test Loss after fine tune: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                acc_finetune = 100. * correct / total
                print('Encode ratio: %.3f | partition index: %d | QF: %d' %(encoded_data_size/raw_data_size, index, QF))
                ratio_encode = encoded_data_size/raw_data_size

                encode_data = {
                    'original acc': original_acc,
                    'acc': acc,
                    'acc_finetune': acc_finetune,
                    'partition index': index,
                    'QF': QF,
                    'encode ratio': ratio,
                    'encode ratio finetune': ratio_encode
                }
                data.append(encode_data)

        with open('./test_encode.json', 'w') as fp:
            json.dump(data, fp, indent=2)

    if args.test_pruned:
        use_cuda = 0
        cfg = pruner.get_cfg()
        conv_index_max = pruner.get_conv_index_max()
        original_data = []
        prune_data = []
        prune_layer_data = []
        prune_layer_test_accuracy_data = []

        last_conv_index = 0  # log for checkpoint restoring, nearest conv layer
        for index in range(len(cfg)):
            # original
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.train')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)
            data = test(index)
            if data['acc'] == -1:
                data['acc'] = acc
            original_data.append(data)

            # prune
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)
            data = test(index)
            if data['acc'] == -1:
                data['acc'] = acc
            prune_data.append(data)

            # prune_layer
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.prune_layer_' + str(last_conv_index))
            # checkpoint = torch.load('./checkpoint/ckpt.prune')
            net = checkpoint['net']
            acc = checkpoint['acc']
            if use_cuda:
                net.cuda()
                # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                # cudnn.benchmark = True
            # create new pruner in each iteration
            pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
            total_filter_num_pre_prune = pruner.total_num_filters(conv_index=-1)
            data = test(index)
            if data['acc'] == -1:
                data['acc'] = acc
                prune_layer_data.append(data)

            # prune_layer_test_accuracy
            for prune_iteration in range(0, 16):
                print('===> Resuming from checkpoint..')
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                if os.path.exists('./checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration)):
                    checkpoint = torch.load('./checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration))
                    print('./checkpoint/ckpt.prune_layer_' + str(last_conv_index) + '_iteration_' + str(prune_iteration))
                    # checkpoint = torch.load('./checkpoint/ckpt.prune')
                    net = checkpoint['net']
                    acc = checkpoint['acc']
                    if use_cuda:
                        net.cuda()
                    pruner = FilterPruner(net.module if isinstance(net, torch.nn.DataParallel) else net)
                    for i in range(10):
                        data = test(index, prune_iteration)
                        if data['acc'] == -1:
                            data['acc'] = acc
                            prune_layer_test_accuracy_data.append(data)

            if index + 1 < len(cfg):
                if not isinstance(cfg[index + 1], str):
                    last_conv_index += 1

        with open('./log_original.json', 'w') as fp:
                        json.dump(original_data, fp, indent=2)
        with open('./log_prune.json', 'w') as fp:
                        json.dump(prune_data, fp, indent=2)
        with open('./log_prune_layer.json', 'w') as fp:
                        json.dump(prune_layer_data, fp, indent=2)
        with open('./log_prune_layer_test_accuracy.json', 'w') as fp:
            json.dump(prune_layer_test_accuracy_data, fp, indent=2)


