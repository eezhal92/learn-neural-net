"""
Train to predict mean
"""

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()

# 10.5, 5, 9.5, 12 => 18.5

# Add layers to model
# quite deep network
model.add(Dense(8, activation='relu', input_dim=4)) # four dimension for input 
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) # one dimension for output

# wide but shallow network
# model.add(Dense(8, activation='relu', input_dim=4)) 
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

# inputs
x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [10, 12, 9, 13],
  [99, 100, 101, 102],
  [105, 111, 109, 102]
])

# outputs
y_train = np.array([
  [2.5],
  [3.25],
  [10.0],
  [11.0],
  [100.5],
  [106.75],
])

# shuffle data
perm = np.random.permutation(y_train.size)
x_train = x_train[perm]
y_train = y_train[perm]

x_val = np.array([
  [1.5, 4, 3, 2.5],
  [10, 14, 11.5, 12],
  [111, 99, 105, 107]
])

y_val = np.array([
  [2.75],
  [11.875],
  [105.5]
])

model.fit(
  x_train,
  y_train,
  batch_size=2,
  # how many times the network will loop through entire dataset. more epoch, better accuracy
  # Increase epochs if based on layer numbers
  epochs=1000, 
  verbose=1,
  # set validation data manually
  validation_data=(x_val, y_val)
)

# test datasets
x_test = np.array([
  [2, 5, 4.5, 1],
  [9, 16, 11, 10.5],
  [100, 95, 99, 102]
])

y_test = np.array([
  [3.125],
  [11.625],
  [99.0]
])

output = model.evaluate(x_test, y_test)

print("==Evaluation==")
print(model.metrics_names) # metric name
print(output) # loss of unseen data

"""
In a well-trained and well-tuned neural net,
the training loss, validation loss, and test loss should be similar
which means there's no underfitting or overfitting issues
Reserve test set until the end of training and validation to make good performance of neuralnet on the future unseen data

Test different strategy on the datasets because every one of them is different

Steps:
Train & Validate, then test dan evaluate
"""