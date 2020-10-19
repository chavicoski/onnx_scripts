from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext import datasets, data


class Net(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=32, output_dim=1):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Build the model
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.recurrent = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            )

    def forward(self, x):
        embedded = self.embedding(x)  
        # embedded = [n_seq, batch_size, n_embedding]

        lstm_out, _ = self.recurrent(embedded)  
        # lstm_out = [n_seq, batch_size, n_hidden]

        lstm_out = lstm_out[-1]  
        # lstm_out = [batch_size, n_hidden]

        dense_out = self.dense(lstm_out)
        # dense_out = [batch_size, output_dim]

        return dense_out  # Get last pred from seq


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    correct = 0
    current_samples = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = batch.text, batch.label.float()
        target = target.view((target.shape[0], 1))
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        pred = torch.round(output)
        correct += pred.eq(target).sum().item()
        current_samples += len(target)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / current_samples))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch.text, batch.label.float()
            target = target.view((target.shape[0], 1))
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.round(output)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch IMDB LSTM Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='Max size of the vocabulary (default: 2000)')
    parser.add_argument('--output-path', type=str, default="onnx_models/lstm_imdb.onnx", 
                        help='Output path to store the onnx file')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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

    model = Net(input_dim=len(TEXT.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_iterator, criterion, optimizer, epoch)
        test(model, device, test_iterator, criterion)

    # Save to ONNX file
    dummy_input = torch.zeros((1000, args.batch_size)).long().to(device)
    torch.onnx._export(model, dummy_input, args.output_path, keep_initializers_as_inputs=True)

if __name__ == '__main__':
    main()
