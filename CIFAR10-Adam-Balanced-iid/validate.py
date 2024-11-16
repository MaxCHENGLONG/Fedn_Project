import os
import sys
import torch
import numpy as np
import json
from model import load_parameters
from data import load_data
from fedn.utils.helpers.helpers import save_metrics
from datetime import datetime

# 获取当前文件的目录路径
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

def calculate_path_norm(model):
    """计算神经网络的路径范数。

    :param model: PyTorch 模型。
    :type model: torch.nn.Module
    :return: 路径范数值。
    :rtype: float
    """
    path_norm = 0.0
    
    # 遍历模型中的参数
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只考虑权重参数来计算路径范数
            path_norm += torch.sum(param ** 2).item()  # 累加每个权重参数的平方值
    
    # 对平方和取平方根得到路径范数
    path_norm = np.sqrt(path_norm)
    return path_norm

def validate(in_model_path, out_json_path, data_path=None):
    """验证模型。

    :param in_model_path: 输入模型的路径。
    :type in_model_path: str
    :param out_json_path: 保存输出 JSON 的路径。
    :type out_json_path: str
    :param data_path: 数据文件的路径。
    :type data_path: str
    """
    # 加载数据 
    x_train, y_train = load_data(data_path)
    x_test, y_test = load_data(data_path, is_train=False)
    
    # Convert and prepare data
    x_train = torch.FloatTensor(x_train).permute(0, 3, 1, 2) # NHWC -> NCHW
    x_train = x_train / 255.0  # Normalize
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test).permute(0, 3, 1, 2) # NHWC -> NCHW
    x_test = x_test / 255.0  # Normalize
    y_test = torch.LongTensor(y_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    model = load_parameters(in_model_path)
    model = model.to(device)
    model.eval()

    # 自定义步骤：在验证之前计算全局模型的路径范数
    global_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Global Model before Validation: {global_path_norm}")

    # 评估模型
    criterion = torch.nn.CrossEntropyLoss() 
    total_train_loss = 0
    total_train_correct = 0
    total_test_loss = 0
    total_test_correct = 0
    
    # 训练集评估
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_train_loss += loss.item() * batch_x.size(0)
            total_train_correct += torch.sum(torch.argmax(outputs, dim=1) == batch_y).item()
    
    # 测试集评估
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_test_loss += loss.item() * batch_x.size(0)
            total_test_correct += torch.sum(torch.argmax(outputs, dim=1) == batch_y).item()
    
    training_loss = total_train_loss / len(x_train)
    training_accuracy = total_train_correct / len(x_train)
    test_loss = total_test_loss / len(x_test)
    test_accuracy = total_test_correct / len(x_test)
    
    # 输出变量值
    print(f"Training Loss: {training_loss}")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
        # 在验证之后计算本地模型的路径范数
    local_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Local Model after Validation: {local_path_norm}")

    # 添加时间戳
    report = {
        "model_id": os.path.basename(in_model_path),  # 模型唯一标识符
        "training_loss": training_loss,  # 训练损失
        "training_accuracy": training_accuracy,  # 训练准确率
        "test_loss": test_loss,  # 测试损失
        "test_accuracy": test_accuracy,  # 测试准确率
        "global_path_norm": global_path_norm,  # 全局模型路径范数
        "local_path_norm": local_path_norm,  # 本地模型路径范数
        "time": datetime.now().isoformat(),
    }

    # 构造文件名（使用model_id作为文件名）
    client_id = os.environ.get("CLIENT_ID", 1)
    alpha = float(os.environ.get("ALPHA", 10))
    filename = f"report_{client_id}_{alpha}.json"

    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(report)
            else:
                existing_data = [existing_data, report]
            f.seek(0)
            json.dump(existing_data, f, indent=4)
    else:
        with open(filename, 'w') as f:
            json.dump([report], f, indent=4)

    save_metrics(report, out_json_path)

if __name__ == "__main__":
    # 调用验证函数，传入命令行参数
    validate(sys.argv[1], sys.argv[2])
