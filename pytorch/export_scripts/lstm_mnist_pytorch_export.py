from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(28, 32)
        self.lrelu1 = nn.LeakyReLU()
        self.recurrent = nn.LSTM(32, 128)
        self.dense2 = nn.Linear(128, 32)
        self.lrelu2 = nn.LeakyReLU()
        self.dense3 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.lrelu1(x)
        lstm_out, (h, c) = self.recurrent(x)
        h = h.view((64, 128))
        x = self.dense2(h)
        x = self.lrelu2(x)
        x = self.dense3(x)
        out = self.softmax(x)
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    current_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.permute(1, 0, 2)  # seq_len, batch_size, in_len
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        current_samples += data.size(0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.permute(1, 0, 2)  # seq_len, batch_size, in_len
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Conv2D MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str, default="onnx_models/lstm_mnist.onnx", 
                        help='Output path to store the onnx file')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    class remove_ch(object):
        ''' Custom transform to preprocess data'''
        def __call__(self, img):
            return img.view((28, 28))

    transform=transforms.Compose([
        transforms.ToTensor(),
        remove_ch()
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, drop_last=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save to ONNX file
    dummy_input = torch.randn(28, args.batch_size, 28, device=device)
    torch.onnx._export(model, dummy_input, args.output_path, keep_initializers_as_inputs=True)

if __name__ == '__main__':
    main()