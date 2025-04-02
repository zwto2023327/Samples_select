import random
import numpy as np
import os
import torch
import pickle
from PIL import Image
import re
def set_random_seed(seed = 10):
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
        if label == target:
            continue
        img[2, :, :] = (1 - alpha[2, :, :]) * img[2, :, :] + alpha[2, :, :] * trigger[2, :, :]
        img[2, :, :] = torch.clamp(img[2, :, :], 0, 1)
        dataset_.append((img, target))
    return dataset_

def Add_Clean_Label_Train_Trigger(dataset, trigger, target, alpha, class_order):
    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            img[2,:,:] = (1-alpha[2,:,:])*img[2,:,:] + alpha[2,:,:]*trigger[2,:,:]
            img[2,:,:] = torch.clamp(img[2,:,:], 0, 1)
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
#选择最容易潜藏的samples
def Add_Clean_Label_Train_Trigger_Quantize(dataset, target, class_order, num_levels, posion_rate):
    dataset_ = list()
    temp = list()
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    j = 0
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if i in class_order:
            temp_img = torch.zeros_like(img[2, :, :])
            t_img = torch.zeros_like(img[2, :, :])
            t_step = 255 // (24 - 1)
            temp_img = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
            t_img = (((img[2, :, :] * 255) // t_step + 1) * t_step) / 255
            temp_img = torch.clamp(temp_img, 0, 1)
            t_img = torch.clamp(t_img, 0, 1)
            temp.append((img, label, torch.sum(torch.abs(temp_img - t_img)), j))
            j = j + 1
        else:
            dataset_.append((img, data[1], 0))
    n = int(len(class_order) * posion_rate)
    sorted_tuples = sorted(temp, key=lambda x: x[2])
    nlargest = [t[3] for t in sorted_tuples[:n]]
    for i in range(len(temp)):
        if i in nlargest:
            img = temp[i][0]
            img[0, :, :] = (((img[0, :, :] * 255) // step_R + 1) * step_R) / 255
            img[1, :, :] = (((img[1, :, :] * 255) // step_G + 1) * step_G) / 255
            img[2, :, :] = (((img[2, :, :] * 255) // step_B + 1) * step_B) / 255
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, temp[i][1], 1))
        else:
            dataset_.append((temp[i][0], temp[i][1], 0))

    return dataset_

def get_stats(selection, output_dir, epoch, seed):
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
    else:
        raise ValueError('Unknown selection {}'.format(selection))
    return stats_metric, stats_class, stats_order
