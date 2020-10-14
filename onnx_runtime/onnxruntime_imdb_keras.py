import torch
import os
import numpy as np
import onnx
import keras2onnx
import onnxruntime
from onnx2keras import onnx_to_keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

onnx_models_path = "onnx_models"
model_name = "trained_model.onnx"
max_features = 2000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 10

print('Loading data...')
_, (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

def batch_gen(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size].astype(np.float32), y[i:i+batch_size]

# Prepare ONNX runtime
session = onnxruntime.InferenceSession(os.path.join(onnx_models_path, model_name), None)  # Create a session with the onnx model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print('Input name: ', input_name)
print('Output name: ', output_name)
   
# Inference with ONNX runtime
correct = 0
total = 0
for data, labels in batch_gen(x_test, y_test, batch_size):
    output = session.run([output_name], {input_name: data})
    pred = torch.round(torch.tensor(output)).reshape((batch_size))
    correct += pred.eq(torch.tensor(labels)).sum().item()
    total += batch_size

print(f"Results: Accuracy = {(correct/total)*100:.2f}({correct}/{total})")

