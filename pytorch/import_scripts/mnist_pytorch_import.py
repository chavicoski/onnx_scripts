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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Conv2D MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--onnx-models-path', type=str, default="onnx_models",
                    help='Path to the folder to store the onnx models')
parser.add_argument('-m', '--model-filename', type=str, default="conv2D_mnist.onnx",
                    help='Name of the model file')
parser.add_argument('--no-2D-input', action='store_true', default=False,
                    help='To change the input size to a 784 length vector')
parser.add_argument('--channel-last', action='store_true', default=False,
                    help='Change input shape from channel first to channel last')
args = parser.parse_args()

device = torch.device("cpu")

kwargs = {'batch_size': args.batch_size}

class from2Dto1D(object):
    ''' Custom transform to preprocess data'''
    def __call__(self, img):
        return img.view((1, -1))

_transforms = [transforms.ToTensor()]
if args.no_2D_input:
    _transforms.append(from2Dto1D())
transform = transforms.Compose(_transforms)

dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, drop_last=True, **kwargs)

onnx_filepath = os.path.join(args.onnx_models_path, args.model_filename)
print(f"Going to load the ONNX model from \"{onnx_filepath}\"")
model = onnx.load(onnx_filepath)

# Check that the IR is well formed
#onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print("Going to build the caffe2 model from ONNX model")
rep = backend.prepare(model, device="CPU") # or CPU
print("Caffe2 model built!")

# Inference with Caffe2 backend (only way to "import with pytorch")
correct = 0
total = 0
for data, label in tqdm(data_loader):
    data, label = data.numpy(), label.numpy()
    if args.channel_last:
        data = np.reshape(data, (data.shape[0], 1, -1))
    outputs = rep.run(data)
    prediction = np.array(np.argmax(np.array(outputs).squeeze(), axis=1).astype(np.int))
    correct += np.sum(prediction == label)
    total += len(prediction)

print(f"Results: Accuracy = {(correct/total)*100:.2f}({correct}/{total})")
