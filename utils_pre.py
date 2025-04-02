import random
import numpy as np
import os
import torch
import pickle
import re
import math
from PIL import Image
from jupyter_core.version import pattern


def set_random_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label, is_poison = self.data[idx][0], self.data[idx][1], self.data[idx][2]
        if self.transform:
            sample = self.transform(sample)
        return (sample, label, is_poison)


def Add_Test_Trigger(dataset, trigger, target, alpha):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        # todo why
        if label == target:
            continue
        img = (1 - alpha) * img + alpha * trigger
        img = torch.clamp(img, 0, 1)
        dataset_.append((img, target))
    return dataset_


def Add_Clean_Label_Train_Trigger(dataset, trigger, target, alpha, class_order):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img = (1 - alpha) * img + alpha * trigger
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, label, 1))
        else:
            dataset_.append((img, data[1], 0))
    return dataset_


def Add_Test_Trigger_Quantize(dataset, target, num_levels):
    dataset_ = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        img[0, :, :] = (((img[0, :, :] * 255) // step_R + 1) * step_R) / 255
        img[1, :, :] = (((img[1, :, :] * 255) // step_G + 1) * step_G) / 255
        img[2, :, :] = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
        img = torch.clamp(img, 0, 1)
        dataset_.append((img, target))
    return dataset_


def Add_Clean_Label_Train_Trigger_Quantize(dataset, target, class_order, num_levels):
    dataset_ = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img[0, :, :] = (((img[0, :, :] * 255) // step_R + 1) * step_R) / 255
            img[1, :, :] = (((img[1, :, :] * 255) // step_G + 1) * step_G) / 255
            img[2, :, :] = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, label, 1))
        else:
            dataset_.append((img, data[1], 0))
    return dataset_


def get_stats(selection, output_dir, epoch, seed, num_classes, target, res_sel):
    if selection == 'loss':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['loss']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'grad':
        fname = os.path.join(output_dir, 'resnet_loss_grad_epoch_{}_seed_{}.pkl'.format(epoch, seed))
        with open(fname, "rb") as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['grad_norm']
        stats_class = loaded['class']
        stats_order = np.arange(len(stats_metric))
    elif selection == 'forget':
        fname = os.path.join(output_dir, 'stats_forget_seed_{}.pkl'.format(seed))
        with open(fname, 'rb') as fin:
            loaded = pickle.load(fin)
        stats_metric = loaded['forget']
        stats_class = loaded['class']
        stats_order = loaded['original_index']
    elif selection == 'res':
        fname = os.path.join(output_dir, 'stats_forget_seed_{}.pkl'.format(seed))
        with open(fname, 'rb') as fin:
            loaded = pickle.load(fin)
        cls = {}
        res = loaded['res']
        sum = 0.0
        stats_class = loaded['class']
        stats_order = loaded['original_index']
        for ind in range(len(stats_order)):
            index = stats_order[ind]
            if stats_class[ind] == target:
                for i in range(num_classes):
                    cls_res = cls.get(i, 0)
                    res_index = res.get(index, {})
                    value_res = res_index.get(i, 0)
                    cls[i] = cls_res + value_res
                    sum += value_res
        if res_sel == "linear":
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - (cls_res * 1.0 / sum)
        elif res_sel == "max" or res_sel == "num":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + 1.0 * math.exp(-cls_res)
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1.0 * math.exp(-cls_res) / sum
        elif res_sel == "exp":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + 1.0 * math.exp(-cls_res)
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1.0 * math.exp(-cls_res) / sum
        elif res_sel == "log":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + 1.0 * math.log(1 + cls_res)
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - 1.0 * math.log(1 + cls_res) / sum
        elif res_sel == "square":
            sum = 0
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                sum = sum + cls_res * cls_res
            for i in range(num_classes):
                cls_res = cls.get(i, 0)
                cls[i] = 1 - 1.0 * cls_res * cls_res / sum
        stats_metric = []
        if res_sel == "num":
            for index in stats_order:
                index_sum = 0
                for i in range(num_classes):
                    cls_res = cls.get(i, 0)
                    res_index = res.get(index, {})
                    value_res = res_index.get(i, 0)
                    if value_res > 0:
                        index_sum += cls_res
                stats_metric.append(index_sum)
        else:
            for index in stats_order:
                index_sum = 0
                for i in range(num_classes):
                    cls_res = cls.get(i, 0)
                    res_index = res.get(index, {})
                    value_res = res_index.get(i, 0)
                    if res_sel == "max":
                        index_sum = max(index_sum, cls_res * value_res)
                    else:
                        index_sum = index_sum + cls_res * value_res
                stats_metric.append(index_sum)
    else:
        raise ValueError('Unknown selection {}'.format(selection))
    return stats_metric, stats_class, stats_order
