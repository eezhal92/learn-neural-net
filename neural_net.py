"""
Train to predict mean
"""

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()

# 10.5, 5, 9.5, 12 => 18.5

# Add layers to model
model.add(Dense(8, activation='relu', input_dim=4)) # four dimension for input 
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) # one dimension for output

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

model.fit(
  x_train,
  y_train,
  batch_size=2,
  # how many times the network will loop through entire dataset. more epoch, better accuracy
  epochs=100, 
  verbose=1,
  # split training and validation datasets. decimal between 0 and 1
  # it's percentage of traning. 0.2 = 20% validation, 80% training
  validation_split=0.2 
)

"""
When we have not enough data:
When we run it, validation and training losses tend to jump all over the place. Producing very incosistent loss and loss values
It means sometimes the network is overfitting the data and sometimes is underfitting the data

"""