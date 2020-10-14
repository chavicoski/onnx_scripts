from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import caffe2.python.onnx.backend as backend
import onnx
from tqdm import tqdm
from torchtext import datasets, data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch IMDB LSTM Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--onnx-models-path', type=str, default="onnx_models",
                    help='Path to the folder to store the onnx models')
parser.add_argument('-m', '--model-filename', type=str, default="lstm_imdb.onnx",
                    help='Name of the model file')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--vocab-size', type=int, default=2000,
                    help='Max size of the vocabulary (default: 2000)')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cpu")

# Create data fields for preprocessing
TEXT = data.Field()
LABEL = data.LabelField()
# Create data splits
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# Create vocabulary
TEXT.build_vocab(train_data, max_size = args.vocab_size)
LABEL.build_vocab(train_data)
# Create splits iterators
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = args.batch_size,
    device = device)

onnx_filepath = os.path.join(args.onnx_models_path, args.model_filename)
print(f"Going to load the ONNX model from \"{onnx_filepath}\"")
model = onnx.load(onnx_filepath)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print("Going to build the caffe2 model from ONNX model")
rep = backend.prepare(model, device="CPU") # or CPU
print("Caffe2 model built!")

# Inference with Caffe2 backend (only way to "import with pytorch")
correct = 0
total = 0
for batch in tqdm(test_iterator):
    data, label = batch.text.numpy(), batch.label.float()
    outputs = torch.tensor(rep.run(data))
    pred = torch.round(outputs)
    correct += pred.eq(label).sum().item()
    total += len(label)

print(f"Results: Accuracy = {(correct/total)*100:.2f}({correct}/{total})")
