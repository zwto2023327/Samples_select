import os
from skimage import io
import torchvision as tv
import numpy as np
import torch

'''def Cifar100(root):

    character_train = [[] for i in range(100)]
    character_test = [[] for i in range(100)]

    train_set = tv.datasets.CIFAR100(root, train=True, download=True)
    test_set = tv.datasets.CIFAR100(root, train=False, download=True)

    trainset = []
    testset = []
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        trainset.append(list((i, np.array(X), np.array(Y))))
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        testset.append(list((i, np.array(X), np.array(Y))))

    for i, X, Y in trainset:
        character_train[Y].append(list((i, X)))  # 32*32*3

    for i, X, Y in testset:
        character_test[Y].append(list((i, X)))  # 32*32*3

    os.mkdir(os.path.join(root, 'train'))
    os.mkdir(os.path.join(root, 'val'))

    for i, per_class in enumerate(character_train):
        character_path = os.path.join(root, 'train', 'character_' + str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)

    for i, per_class in enumerate(character_test):
        character_path = os.path.join(root, 'val', 'character_' + str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)'''

def Cifar100(root):
    '''os.mkdir(os.path.join(root, 'cifar25'))
    os.mkdir(os.path.join(root, 'cifar25','train'))
    os.mkdir(os.path.join(root, 'cifar25','val'))
    for i in range(25):
        character_path = os.path.join(root, 'cifar25', 'train', 'class_' + str(i))
        os.mkdir(character_path)
        character_path = os.path.join(root, 'cifar25', 'val', 'class_' + str(i))
        os.mkdir(character_path)
    train_set = tv.datasets.CIFAR100(root, train=True, download=True)
    test_set = tv.datasets.CIFAR100(root, train=False, download=True)
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        if Y < 25:
            character_path = os.path.join(root, 'cifar25', 'train', 'class_' + str(Y))
            img = np.array(X)
            img_path = character_path + '/' + str(i) + ".jpg"
            io.imsave(img_path, img)
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        if Y < 25:
            character_path = os.path.join(root, 'cifar25', 'val', 'class_' + str(Y))
            img = np.array(X)
            img_path = character_path + '/' + str(i) + ".jpg"
            io.imsave(img_path, img)
    os.mkdir(os.path.join(root, 'cifar50'))
    os.mkdir(os.path.join(root, 'cifar50', 'train'))
    os.mkdir(os.path.join(root, 'cifar50', 'val'))
    num = {}
    for i in range(50):
        num[i] = 0
        character_path = os.path.join(root, 'cifar50', 'train', 'class_' + str(i))
        os.mkdir(character_path)
        character_path = os.path.join(root, 'cifar50', 'val', 'class_' + str(i))
        os.mkdir(character_path)
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        if Y < 50 :
            if Y == 0 or (num[Y] < 245):
                num[Y] = num[Y] + 1
                character_path = os.path.join(root, 'cifar50', 'train', 'class_' + str(Y))
                img = np.array(X)
                img_path = character_path + '/' + str(i) + ".jpg"
                io.imsave(img_path, img)
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        if Y < 50 :
            if Y == 0 or (num[Y] < 50):
                character_path = os.path.join(root, 'cifar50', 'val', 'class_' + str(Y))
                img = np.array(X)
                img_path = character_path + '/' + str(i) + ".jpg"
                io.imsave(img_path, img)
    os.mkdir(os.path.join(root, 'cifar75'))
    os.mkdir(os.path.join(root, 'cifar75', 'train'))
    os.mkdir(os.path.join(root, 'cifar75', 'val'))
    num = {}
    for i in range(75):
        num[i] = 0
        character_path = os.path.join(root, 'cifar75', 'train', 'class_' + str(i))
        os.mkdir(character_path)
        character_path = os.path.join(root, 'cifar75', 'val', 'class_' + str(i))
        os.mkdir(character_path)
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        if Y < 75:
            if Y == 0 or (num[Y] < 163):
                num[Y] = num[Y] + 1
                character_path = os.path.join(root, 'cifar75', 'train', 'class_' + str(Y))
                img = np.array(X)
                img_path = character_path + '/' + str(i) + ".jpg"
                io.imsave(img_path, img)
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        if Y < 75:
            if Y == 0 or (num[Y] < 33):
                character_path = os.path.join(root, 'cifar75', 'val', 'class_' + str(Y))
                img = np.array(X)
                img_path = character_path + '/' + str(i) + ".jpg"
                io.imsave(img_path, img)'''
    os.mkdir(os.path.join(root, 'cifar100'))
    os.mkdir(os.path.join(root, 'cifar100', 'train'))
    os.mkdir(os.path.join(root, 'cifar100', 'val'))
    train_set = tv.datasets.CIFAR100(root, train=True, download=True)
    test_set = tv.datasets.CIFAR100(root, train=False, download=True)
    num = {}
    for i in range(100):
        num[i] = 0
        character_path = os.path.join(root, 'cifar100', 'train', 'class_' + str(i))
        os.mkdir(character_path)
        character_path = os.path.join(root, 'cifar100', 'val', 'class_' + str(i))
        os.mkdir(character_path)
    for i, (X, Y) in enumerate(train_set):  # 将train_set的数据和label读入列表
        if Y == 0 or (num[Y] < 120):
            num[Y] = num[Y] + 1
            character_path = os.path.join(root, 'cifar100', 'train', 'class_' + str(Y))
            img = np.array(X)
            img_path = character_path + '/' + str(i) + ".jpg"
            io.imsave(img_path, img)
    for i, (X, Y) in enumerate(test_set):  # 将test_set的数据和label读入列表
        if Y == 0 or (num[Y] < 24):
            character_path = os.path.join(root, 'cifar100', 'val', 'class_' + str(Y))
            img = np.array(X)
            img_path = character_path + '/' + str(i) + ".jpg"
            io.imsave(img_path, img)
if __name__ == '__main__':
    root = '/home/boot/STU/DATASETS/EQU_CIFAR'
    Cifar100(root)




