import argparse
import numpy as np
import time
import os
import torch
import re
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import heapq
import logging
from cifar_resnet import ResNet18, ResNet50, ResNet34
from utils_B import *

def train_step(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels, is_poison) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def test_step(model, criterion, data_loader, target):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    target_correct = 0
    target_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            for i in range(len(pred)):
                a = pred[i].long()
                b = labels.data.view_as(pred)[i].long()
                if b == target:
                    if a == b:
                        target_correct += 1
                    target_num += 1
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    if target_num != 0:
        tar_acc = float(target_correct) / target_num
    else:
        tar_acc = 0
    return loss, acc, tar_acc

def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"

parser = argparse.ArgumentParser(description='Evaluate backdoor attack with different selection methods')
parser.add_argument('--model', default='resnet34', choices=['resnet18', 'resnet50', 'resnet34'])
parser.add_argument('--selection', default='forget', choices=['random', 'loss', 'grad', 'forget'])
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1500, help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--output_dir', type=str, default='save_metric', help='directory where to save metrics, same with the one used in cal_metric.py')
parser.add_argument('--result_dir', type=str, default='test', help='directory where to save results')
parser.add_argument('--model_dir', type=str, default='/home/boot/STU/workspaces/wzx/Samples_select/models/', help='directory where to save results')
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--num_levels', type=str, default="36:60:12")
parser.add_argument('--poison_rate', type=float, default=0.33)
parser.add_argument('--all_rate', type=float, default=0.99)
parser.add_argument('--backdoor_type', default='quantize', choices=['badnets', 'blend', 'quantize'])
parser.add_argument('--select_epoch', type=int, default=10, help='epoch which to calculate the stats')
args = parser.parse_args()
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
set_random_seed(args.seed)

if args.dataset == 'cifar10':
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 10
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar100':
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 100
    test_dataset = datasets.CIFAR100(root='./data100', train=False, transform=transforms.ToTensor(), download=True)


if args.backdoor_type == 'badnets':
    checkboard = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]]).repeat((3,1,1))
    trigger = torch.zeros([3, 32, 32])
    trigger[:, 24:29, 15:20] = 1
    trigger_alpha = torch.zeros([3, 32, 32])
    trigger_alpha[:, 24:29, 15:20] = 1.0
elif args.backdoor_type == 'blend':
    trigger = np.load('/home/boot/STU/workspaces/wzx/bench/resource/blended/hello_kitty_32.npy')
    trigger = torch.from_numpy(trigger)
    trigger = np.transpose(trigger, (2, 0, 1))
    trigger = trigger / 255
    trigger = trigger.type(torch.FloatTensor)
    trigger_alpha = torch.ones([3, 32, 32])
    trigger_alpha *= 0.3
#total_poison = int(len(train_dataset) * args.poison_rate)
if args.selection in ['loss', 'grad', 'forget']:
    stats_metric, stats_class, stats_inds = get_stats(args.selection, args.output_dir, args.select_epoch, args.seed)
    metric_vals, metric_inds = [], []
    #只投毒了target-label数据
    for i in range(len(train_dataset)):
        if stats_class[i] == args.y_target:
            metric_vals.append(stats_metric[i])
            metric_inds.append(stats_inds[i])
    #largest_inds = heapq.nlargest(total_poison, range(len(metric_vals)), metric_vals.__getitem__)
    if args.backdoor_type == 'quantize':
        total_poison = int(len(metric_vals) * args.all_rate)
    else:
        total_poison = int(len(metric_vals) * args.poison_rate)
    largest_inds = heapq.nlargest(total_poison, range(len(metric_vals)), metric_vals.__getitem__)
    poison_inds = [metric_inds[i] for i in largest_inds]
else:
    shuffle = np.random.permutation(len(train_dataset))
    k = 0
    poison_inds = []
    total_poison = 500 * args.poison_rate
    for i in shuffle:
        if train_dataset[i][1] == args.y_target and k < total_poison:
            poison_inds.append(i)
            k += 1
if args.backdoor_type == 'quantize':
    poison_train_set = Add_Clean_Label_Train_Trigger_Quantize(train_dataset, args.y_target, poison_inds, args.num_levels, args.poison_rate)
    poison_test_set = Add_Test_Trigger_Quantize(test_dataset, args.y_target, args.num_levels)
else:
    poison_train_set = Add_Clean_Label_Train_Trigger(train_dataset, trigger, args.y_target, trigger_alpha, poison_inds)
    poison_test_set = Add_Test_Trigger(test_dataset, trigger, args.y_target, trigger_alpha)
poison_train_set = MyDataset(poison_train_set, train_transform)
train_loader = DataLoader(poison_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
trigger_loader = DataLoader(poison_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif args.model == 'resnet34':
    model = ResNet34(num_classes=num_classes)
model = model.cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss().to(device)

if args.dataset == 'cifar10':
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[60, 90], gamma=0.1)
else:
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=False,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[150, 225], gamma=0.1)

os.makedirs(args.result_dir, exist_ok=True)
logger = logging.getLogger()
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])
logger.info(args)
logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC \t TargetACC \t CleanTargetACC')
for epoch in range(args.epochs):
    start = time.time()
    lr = model_optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_step(model, criterion, model_optimizer, train_loader)
    cl_test_loss, cl_test_acc, cl_tar_acc= test_step(model, criterion, test_loader, args.y_target)
    po_test_loss, po_test_acc, po_tar_acc = test_step(model, criterion, trigger_loader, args.y_target)
    if epoch%100 == 0:
        path = os.path.join(args.model_dir, args.result_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        path_to_save = os.path.join(args.model_dir, args.result_dir, str(epoch)+".pt")
        torch.save(model.state_dict(), path_to_save)
    scheduler.step()
    end = time.time()
    logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc, po_tar_acc, cl_tar_acc)


