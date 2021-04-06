from __future__ import print_function
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Dummy_datagen:
    def __init__(self, batch_size=2):
        # Shape: (n_samples=2, ch=2, depth=8, height=8, width=8)
        self.samples = np.arange(1, (2*3*8*8*8)+1).reshape((2, 3, 8, 8, 8)).astype(np.float32)
        # Shape: (n_samples=2, dim=2)
        self.labels = np.arange(1, (2*2)+1).reshape((2, 2)).astype(np.float32)
        self.curr_idx = 0  # Current index of the batch
        self.bs = batch_size  # Batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.samples.shape[0] / self.bs)

    def __next__(self):
        target = self.curr_idx
        self.curr_idx += self.bs
        if target <= self.samples.shape[0]-self.bs:
            return self.samples[target:target+self.bs], self.labels[target:target+self.bs]
        self.curr_idx = 0  # Reset
        raise StopIteration


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        n_features = -1
        return torch.reshape(x, (batch_size, n_features))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 5, (3, 3, 3), (1, 1, 1), padding=(1, 2, 2)),
            nn.MaxPool3d((2, 2, 2), (1, 2, 2)),
            nn.Conv3d(5, 10, (3, 3, 3), (1, 1, 1), padding=(2, 2, 2)),
            nn.AvgPool3d((2, 2, 2), (2, 2, 2)),
            Flatten(),
            nn.Linear(360, 100),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        return self.model(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    current_samples = 0
    loss_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = data.to(device), target.to(device)
        data_el_size = 1
        for dim in data.size()[1:]:
            data_el_size *= dim
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction='sum')
        loss.backward()
        loss_acc += loss.item() / data_el_size
        current_samples += data.size(0)
        optimizer.step()
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.samples),
            100. * batch_idx / len(train_loader), loss_acc / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    current_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = torch.from_numpy(data), torch.from_numpy(target)
            data, target = data.to(device), target.to(device)
            data_el_size = 1
            for dim in data.size()[1:]:
                data_el_size *= dim
            output = model(data)
            print("test output: ", output)
            test_loss += F.mse_loss(output, target,
                                    reduction='sum').item() / data_el_size
            current_samples += data.size(0)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss / current_samples))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Conv3D synthetic Example')

    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str, default="onnx_models/conv3D_synthetic.onnx",
                        help='Output path to store the onnx file')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create data generators
    train_loader = Dummy_datagen(args.batch_size)
    test_loader = Dummy_datagen(args.batch_size)

    # Train
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 3, 8, 8, 8, device=device)
    torch.onnx._export(model, dummy_input, args.output_path,
                       keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()
