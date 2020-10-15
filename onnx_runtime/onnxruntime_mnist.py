from __future__ import print_function
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms
import onnxruntime

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    parser.add_argument('--input-1D', action='store_true', default=False,
                        help='To change the input size to a 784 length vector')
    parser.add_argument('--no-channel', action='store_true', default=False,
                        help='If --input-1D is enabled, removes the channel dimension. (bs, 1, 784) -> (bs, 784)')
    parser.add_argument('--channel-last', action='store_true', default=False,
                        help='Change input shape from channel first to channel last')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'shuffle': True},
                     )

    # Prepare data loader
    transform=transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, drop_last=True, **kwargs)

    # Prepare ONNX runtime
    session = onnxruntime.InferenceSession(args.onnx_file, None)  # Create a session with the onnx model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
       
    # Inference with ONNX runtime
    correct = 0
    total = 0
    for data, label in tqdm(data_loader):
        data, label = data.numpy(), label.numpy()
        if args.channel_last:
            data = np.reshape(data, (data.shape[0], 28, 28, 1))
        if args.input_1D:
            if args.no_channel:
                data = np.reshape(data, (data.shape[0], -1))
            else:
                data = np.reshape(data, (data.shape[0], 1, -1))
        result = session.run([output_name], {input_name: data})
        prediction = np.array(np.argmax(np.array(result).squeeze(), axis=1).astype(np.int))
        correct += np.sum(prediction == label)
        total += len(prediction)

    print(f"Results: Accuracy = {(correct/total)*100:.2f}({correct}/{total})")

if __name__ == '__main__':
    main()
