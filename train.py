import math
import os
import sys
import csv
import json
import torch
import uuid
from model import load_parameters, save_parameters
from datetime import datetime

from data import load_data
from fedn.utils.helpers.helpers import save_metadata
from validate import calculate_path_norm

# 获取当前文件的目录路径
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=1, lr=0.01):
    """完成模型更新。

    从 in_model_path由 FEDn 客户端管理）加载模型参数，
    执行模型更新，并将更新后的参数写入 out_model_path由 FEDn 客户端获取）。

    :param in_model_path: 输入模型的路径。
    :type in_model_path: str
    :param out_model_path: 保存输出模型的路径。
    :type out_model_path: str
    :param data_path: 数据文件的路径。
    :type data_path: str
    :param batch_size: 使用的批大小。
    :type batch_size: int
    :param epochs: 训练的轮数。
    :type epochs: int
    :param lr: 使用的学习率。
    :type lr: float
    """
    # 加载数据
    x_train, y_train = load_data(data_path)
    client_data_size = len(x_train)

    # 加载参数并初始化模型
    model = load_parameters(in_model_path)

    # # 生成唯一的模型 ID
    # model_id = str(uuid.uuid4())

    # 自定义步骤：在训练之前计算全局模型的路径范数
    global_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Global Model before Training: {global_path_norm}")

    # 训练模型
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    n_batches = int(math.ceil(len(x_train) / batch_size))
    criterion = torch.nn.NLLLoss()
    for e in range(epochs):  # 轮数循环
        for b in range(n_batches):  # 批次循环
            # 获取当前批次的数据
            batch_x = x_train[b * batch_size : (b + 1) * batch_size]
            batch_y = y_train[b * batch_size : (b + 1) * batch_size]
            # 对批次数据进行训练
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # 记录日志
            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # 自定义步骤：在训练之后计算本地模型的路径范数
    local_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Local Model after Training: {local_path_norm}")
    model_id = os.path.basename(out_model_path)
    # 聚合服务器端所需的元数据
    metadata = {
        # num_examples 是必需的
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "global_path_norm": global_path_norm,
        "local_path_norm": local_path_norm,
        "client_data_size": client_data_size,
        "time": datetime.now().isoformat(),
        "model_id": model_id,
    }
    client_id = os.environ.get("CLIENT_ID",1)
    alpha = float(os.environ.get("ALPHA", 10))
    filename = f"metadata_{client_id}_{alpha}.json"
    # 将 metadata 追加保存到 JSON 文件
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(metadata)
            else:
                existing_data = [existing_data, metadata]
            f.seek(0)
            json.dump(existing_data, f, indent=4)
    else:
        with open(filename, 'w') as f:
            json.dump([metadata], f, indent=4)
    # 保存 JSON 格式的元数据文件（必需）
    save_metadata(metadata, out_model_path)

    # 保存模型更新（必需）
    save_parameters(model, out_model_path)

if __name__ == "__main__":
    # 调用训练函数，传入命令行参数
    train(sys.argv[1], sys.argv[2])
