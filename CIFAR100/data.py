import os
from math import floor

import torch
import torchvision
import random
from collections import defaultdict
import numpy as np
# 获取当前文件的目录路径
dir_path = os.path.dirname(os.path.realpath(__file__))
# 获取目录的绝对路径
abs_path = os.path.abspath(dir_path)

def get_data(out_dir="data"):
    # 如果目标目录不存在，则创建目录
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 如果数据集未下载，则下载训练集和测试集
    if not os.path.exists(f"{out_dir}/train"):
        torchvision.datasets.CIFAR100(root=f"{out_dir}/train", transform=torchvision.transforms.ToTensor(), train=True, download=True)
    if not os.path.exists(f"{out_dir}/test"):
        torchvision.datasets.CIFAR100(root=f"{out_dir}/test", transform=torchvision.transforms.ToTensor(), train=False, download=True)

def load_data(data_path, is_train=True):
    """从磁盘加载数据。

    :param data_path: 数据文件的路径。
    :type data_path: str
    :param is_train: 是否加载训练数据。
    :type is_train: bool
    :return: 数据和标签的元组。
    :rtype: tuple
    """
    # 如果未提供数据路径，则从环境变量或默认路径获取
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/cifar.pt")

    # 加载数据
    data = torch.load(data_path, weights_only=True)

    # 根据标志选择训练集或测试集
    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # 归一化处理
    X = X / 255

    return X, y

def splitset_by_dirichlet_imbalanced(dataset, num_clients, alpha, imbalance_ratios=None):
    # 初始化每个客户端的数据容器
    client_data = {i: {'x': [], 'y': []} for i in range(num_clients)}

    # 获取每个类别的数据
    label_data = defaultdict(list)
    for img, label in zip(dataset.data, dataset.targets):
        label_data[label.item()].append((img, label))

    # 使用Dirichlet分布为每个客户端分配数据（非IID）
    for label, items in label_data.items():
        total_items = len(items)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * total_items).astype(int)

        # 分配样本给每个客户端
        start_idx = 0
        for client_id, num_items in enumerate(proportions):
            client_data[client_id]['x'].extend([item[0] for item in items[start_idx:start_idx + num_items]])
            client_data[client_id]['y'].extend([item[1] for item in items[start_idx:start_idx + num_items]])
            start_idx += num_items

    # 调整客户端数据总量使之不平衡
    if imbalance_ratios is not None:
        for client_id in range(num_clients):
            if imbalance_ratios[client_id] < 1.0:
                # 减少客户端数据量
                num_samples = int(len(client_data[client_id]['x']) * imbalance_ratios[client_id])
                indices = np.random.choice(len(client_data[client_id]['x']), num_samples, replace=False)
                client_data[client_id]['x'] = [client_data[client_id]['x'][i] for i in indices]
                client_data[client_id]['y'] = [client_data[client_id]['y'][i] for i in indices]

    # 将数据转换为张量
    for i in range(num_clients):
        if client_data[i]['x']:
            client_data[i]['x'] = torch.stack(client_data[i]['x'])
            client_data[i]['y'] = torch.tensor(client_data[i]['y'])
        else:
            client_data[i]['x'] = torch.tensor([])
            client_data[i]['y'] = torch.tensor([])

    return client_data

def split(out_dir="data"):
    # 从环境变量获取客户端数量和 alpha 值，如果未设置则提供默认值
    num_clients = int(os.environ.get("CLIENT_NUM", 2))  # 默认值为2
    alpha = float(os.environ.get("ALPHA", 10))  # 默认值为10
    
    # 从环境变量获取不平衡比例，如果没有设置，使用默认值
    imbalance_ratios_str = os.environ.get("IMBALANCE_RATIOS", None)
    if imbalance_ratios_str is None:
        # 默认每个客户端均分
        imbalance_ratios = [1.0] * num_clients
    else:
        # 将字符串转换为浮点数列表
        imbalance_ratios = list(map(float, imbalance_ratios_str.split(",")))

    # 创建客户端数据目录
    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    # 加载训练和测试数据集
    train_data = torchvision.datasets.CIFAR100(root=f"{out_dir}/train", download=True, transform=torchvision.transforms.ToTensor(), train=True)
    test_data = torchvision.datasets.CIFAR100(root=f"{out_dir}/test", download=True, transform=torchvision.transforms.ToTensor(), train=False)

    # 使用Dirichlet分布进行数据划分，加入不平衡的条件
    train_splits = splitset_by_dirichlet_imbalanced(train_data, num_clients, alpha, imbalance_ratios)
    test_splits = splitset_by_dirichlet_imbalanced(test_data, num_clients, alpha, imbalance_ratios)

    # 将数据保存到不同的客户端子目录中
    for i in range(num_clients):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": train_splits[i]['x'],
                "y_train": train_splits[i]['y'],
                "x_test": test_splits[i]['x'],
                "y_test": test_splits[i]['y'],
            },
            f"{subdir}/cifar.pt",
        )


if __name__ == "__main__":
    # 如果数据尚未准备好，则准备数据
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data()
        split()
