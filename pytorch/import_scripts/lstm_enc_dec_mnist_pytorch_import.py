from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import caffe2.python.onnx.backend as backend
import onnx
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MNIST LSTM enc dec ONNX import example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--model-path', type=str, default="onnx_models/lstm_enc_dec_mnist.onnx", 
                    help='Path of the onnx file to load')
args = parser.parse_args()

device = torch.device("cpu")

kwargs = {'batch_size': args.batch_size}

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, drop_last=False, **kwargs)

print(f"Going to load the ONNX model from \"{args.model_path}\"")
model = onnx.load(args.model_path)

# Check that the IR is well formed
#onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print("Going to build the caffe2 model from ONNX model")
rep = backend.prepare(model, device="CPU") # or CPU
print("Caffe2 model built!")

# Inference with Caffe2 backend (only way to "import with pytorch")
loss = 0
total_samples = 0
to_torch = lambda x: torch.from_numpy(x)
for data, label in tqdm(data_loader):
    data, label = data.numpy(), label.numpy()  # Caffe2 backend works with numpy not torch.Tensor
    data = np.reshape(data, (data.shape[0], 28, 28))
    outputs = rep.run(data)._0
    loss += ((data - outputs)**2).sum() / (28*28)
    total_samples += data.shape[0]

print(f"Results: mse loss = {(loss/total_samples):.5f}")
