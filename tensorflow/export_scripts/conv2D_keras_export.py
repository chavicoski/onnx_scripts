import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import keras2onnx

# Config
onnx_models_path = "onnx_models"
model_name = "conv2D_mnist"
num_classes = 10
batch_size = 100
epochs = 5

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Get one hot encoding from labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

model = Sequential()
model.add(Input(shape=(28, 28, 1), name="linput"))
model.add(Conv2D(16, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(num_classes, activation = 'softmax'))

model.build(input_shape=(28, 28, 1))  # For keras2onnx 

model.compile(loss = 'categorical_crossentropy', 
        optimizer = "adam",               
        metrics = ['accuracy'])

model.summary()

# Training
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Evaluation
acc = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", acc[0], " Accuracy:", acc[1])

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, model_name, debug_mode=1)
# Save ONNX to file
keras2onnx.save_model(onnx_model, f"{os.path.join(onnx_models_path, model_name)}.onnx")
