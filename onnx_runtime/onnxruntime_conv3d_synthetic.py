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
    parser = argparse.ArgumentParser(description='Test 3d models with synthetic data')

    parser.add_argument('-b', '--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    args = parser.parse_args()

    # Load ONNX model
    onnx_model = onnx.load(args.onnx_file)

    # Prepare ONNX runtime
    # Create a session with the onnx model
    session = onnxruntime.InferenceSession(args.onnx_file, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Prepare data loader
    class Dummy_datagen:
        def __init__(self, batch_size=2):
            # Shape: (n_samples=2, ch=2, depth=8, height=8, width=8)
            self.samples = np.arange(1, (2*3*8*8*8)+1).reshape((2, 3, 8, 8, 8)).astype(np.float32)
            # Shape: (n_samples=2, dim=2)
            self.labels = np.arange(1, (2*2)+1).reshape((2, 2)).astype(np.float32)
            self.curr_idx = 0  # Current index of the batch
            self.bs = batch_size

        def __iter__(self):
            return self

        def __len__(self):
            return int(self.samples.shape[0] / self.bs)

        def __next__(self):
            target = self.curr_idx
            self.curr_idx += self.bs
            if target <= self.samples.shape[0]-self.bs:
                return self.samples[target:target+self.bs], self.labels[target:target+self.bs]
            raise StopIteration

    total_mse = 0
    total_samples = 0
    preds_sum = 0
    for data, label in tqdm(Dummy_datagen(args.batch_size)):
        # Run model
        result = session.run([output_name], {input_name: data})
        pred = np.squeeze(np.array(result), axis=0)
        print(pred)
        preds_sum += pred.sum()
        total_mse += ((label - pred)**2).sum() / (label.size/label.shape[0])
        total_samples += len(data)

    print(f"Results: mse = {(total_mse/total_samples):.5f}")
    print(f"Results: sum = {preds_sum}")


if __name__ == '__main__':
    main()
