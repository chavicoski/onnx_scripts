from __future__ import print_function
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms
import onnxruntime
from torchtext import datasets, data

# Training settings
parser = argparse.ArgumentParser(description='Inference with ONNX runtime from Pytorch ONNX model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                    help='File path to the onnx file with the pretrained model to test')
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

# Prepare ONNX runtime
session = onnxruntime.InferenceSession(args.onnx_file, None)  # Create a session with the onnx model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
   
# Inference with ONNX runtime
correct = 0
total = 0
for batch in tqdm(test_iterator):
    data, label = batch.text.numpy().astype(np.longlong), batch.label.float()
    dummy = np.zeros((1000, args.batch_size))
    try:
        dummy[:data.shape[0], :data.shape[1]] = data
    except:
        continue
    data = dummy.astype(np.longlong)
    output = session.run([output_name], {input_name: data})
    pred = torch.round(torch.tensor(output))
    correct += pred.eq(label).sum().item()
    total += len(label)

print(f"Results: Accuracy = {(correct/total)*100:.2f}({correct}/{total})")
