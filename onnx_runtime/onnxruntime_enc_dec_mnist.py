from __future__ import print_function
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms
import onnx
import onnxruntime

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST recurrent encoder-decoder')
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

    device = torch.device("cpu")

    kwargs = {'batch_size': args.batch_size}

    # Prepare data loader
    transform=transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    # Print ONNX graph
    onnx_model = onnx.load(args.onnx_file)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Prepare ONNX runtime
    session = onnxruntime.InferenceSession(args.onnx_file, None)  # Create a session with the onnx model
    enc_input_name = session.get_inputs()[0].name
    dec_input_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    # Inference with ONNX runtime
    total_mse = 0
    total_samples = 0
    mse_func = lambda x, y: ((x - y)**2).mean()
    for data, label in tqdm(data_loader):
        # Prepare data
        data, label = data.numpy(), label.numpy()
        data = np.reshape(data, (data.shape[0], 28, 28))
        data_shifted = np.pad(data, ((0,0), (1,0), (0,0)), 'constant')[:,0:28,:]
        # Run model
        result = session.run([output_name], {enc_input_name: data, dec_input_name: data_shifted})
        pred = np.squeeze(np.array(result), axis=0)
        total_mse += sum([mse_func(x, y) for x, y in zip(data, pred)])
        total_samples += len(data)

    print(f"Results: mse = {(total_mse/total_samples):.5f}")

if __name__ == '__main__':
    main()
