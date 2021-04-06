from __future__ import print_function
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import onnx
import onnxruntime


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ONNX RUNTIME MNIST Inference')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    parser.add_argument('--input-1D', action='store_true', default=False,
                        help='To change the input size to a 784 length vector')
    parser.add_argument('--no-channel', action='store_true', default=False,
                        help='If --input-1D is enabled, removes the channel dimension. (bs, 1, 784) -> (bs, 784)')
    parser.add_argument('--channel-last', action='store_true', default=False,
                        help='Change input shape from channel first to channel last')
    parser.add_argument('--sequence', action='store_true', default=False,
                        help='To change the input shape to a sequence (seq_len=28, bs, in_len=28)')
    args = parser.parse_args()

    kwargs = {'batch_size': args.batch_size}

    # Prepare data loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, drop_last=True, **kwargs)

    # Print ONNX graph
    onnx_model = onnx.load(args.onnx_file)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Prepare ONNX runtime
    # Create a session with the onnx model
    session = onnxruntime.InferenceSession(args.onnx_file, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Inference with ONNX runtime
    correct = 0
    total = 0

    for data, label in tqdm(data_loader):
        data, label = data.numpy(), label.numpy()
        if args.sequence:
            # Prepare the images as a sequence of length 28
            data = np.reshape(data, (data.shape[0], 28, 28))
            data = np.transpose(data, (1, 0, 2)) # to (seq, batch_size, dim)
        else:
            # Set the shape of the data
            if args.channel_last:
                data = np.reshape(data, (data.shape[0], 28, 28, 1))
            if args.input_1D:
                if args.no_channel:
                    data = np.reshape(data, (data.shape[0], -1))
                else:
                    data = np.reshape(data, (data.shape[0], 1, -1))
            elif args.no_channel:
                data = np.reshape(data, (data.shape[0], 28, 28))

        # Perform inference
        result = session.run([output_name], {input_name: data})

        if args.batch_size == 1:
            prediction = np.array([np.argmax(result).astype(np.int)])
        else:
            prediction = np.array(
                np.argmax(np.array(result).squeeze(), axis=1).astype(np.int))

        correct += np.sum(prediction == label)
        total += len(prediction)

    print(f"Results: Accuracy = {(correct/total)*100:.4f}({correct}/{total})")


if __name__ == '__main__':
    main()
