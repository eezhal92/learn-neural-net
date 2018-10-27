"""
Train to predict mean
Classification
"""

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

model = Sequential()

# 10.5, 5, 9.5, 12 => low

# < 50 = low
# > 50 = high

# Add layers to model
# quite deep network
model.add(Dense(8, activation='relu', input_dim=4)) # four dimension for input 
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid is a function to return single number between 0 and 1

# 0 = low
# 1 = high

model.compile(
  optimizer='adam', 
  loss='binary_crossentropy', # optimize data for being a part of one of two classes
  metrics=['accuracy']
)

x_train = np.array([
  [1, 2, 3, 4],
  [4, 6, 1, 2],
  [10, 9, 10, 11],
  [101, 96, 89, 111],
  [99, 100, 101, 102],
  [105, 111, 109, 102],
])

y_train = np.array([
  [0],
  [0],
  [0],
  [1],
  [1],
  [1]
])

x_val = np.array([
  [1.5, 4, 3, 2.5],
  [10, 14, 11.5, 12],
  [111, 102, 98, 100]
])

y_val = np.array([
  [0],
  [0],
  [1]
])

model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_data=(x_val, y_val)
)