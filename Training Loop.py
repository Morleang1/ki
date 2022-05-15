import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model =keras.Sequential([layers.Dense(64, activation= "relu"),
                          layers.Dense(10)])

### Hyperparameter setting and optimization ###
batch_size = 32
num_epochs = 10
learning_rate = 5e-4

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=len(x_test)).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(x, y):

  with tf.GradientTape() as tape:

    y_hat = model(x)

    loss = loss_func(y,y_hat)

  grads = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  acc_metric.update_state(y, y_hat)
  accuracy =  acc_metric.result()
  return loss, accuracy


@tf.function
def test_step(x,y):
    y_hat = model(x, training=False)
    loss = loss_func(y, y_hat)
    acc_metric.update_state(y, y_hat)
    accuracy = acc_metric.result()
    return loss , accuracy

for epoch in tqdm(range(num_epochs)):
    print("\nEpoch [%d/%d]" % (epoch+1,num_epochs),)

    for (x_batch_train, y_batch_train) in train_dataset:
        loss , accuracy = train_step(x_batch_train, y_batch_train)

    print("training loss: " + str(np.mean(loss)) ," - traning accuracy: " + str(accuracy.numpy()))

for (x_batch_test, y_batch_test) in test_dataset:
    loss , accuracy = test_step(x_batch_test, y_batch_test)

print('test  - loss: ' + str(np.mean(loss)) , '-  accuracy: ' + str(accuracy.numpy()))
