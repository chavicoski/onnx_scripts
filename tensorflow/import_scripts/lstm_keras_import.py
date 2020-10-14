import os
import onnx
import keras2onnx
from onnx2keras import onnx_to_keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

onnx_models_path = "onnx_models"
model_name = "lstm_imdb"
max_features = 2000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 64

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Load ONNX model...')
onnx_model = onnx.load(f"{os.path.join(onnx_models_path, model_name)}.onnx")

#onnx.checker.check_model(onnx_model)

print('Convert ONNX to Keras...')
k_model = onnx_to_keras(onnx_model, ['embedding_input'])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

loss, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print("Evaluation result: Loss:", loss, " Accuracy:", acc)
