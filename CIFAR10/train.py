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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and model
    x_train, y_train = load_data(data_path)
    client_data_size = len(x_train)
    model = load_parameters(in_model_path)
    model = model.to(device)
    
    # Calculate initial path norm
    global_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Global Model before Training: {global_path_norm}")

    # Convert and prepare data
    x_train = torch.FloatTensor(x_train).permute(0, 3, 1, 2) # NHWC -> NCHW
    x_train = x_train / 255.0  # Normalize
    y_train = torch.LongTensor(y_train)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    n_batches = len(train_loader)

    # Training loop
    model.train()
    for e in range(epochs):
        for b, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

    # Calculate final path norm and save metadata
    local_path_norm = calculate_path_norm(model)
    print(f"Path-norm of the Local Model after Training: {local_path_norm}")
    model_id = os.path.basename(out_model_path)
    
    metadata = {
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

    # Save metadata
    client_id = os.environ.get("CLIENT_ID", 1)
    alpha = float(os.environ.get("ALPHA", 10))
    filename = f"metadata_{client_id}_{alpha}.json"
    
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

    save_metadata(metadata, out_model_path)
    save_parameters(model, out_model_path)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])