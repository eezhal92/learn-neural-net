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

data = np.genfromtxt('highlow.csv', delimiter=',')
x_train = data[1:, :4]
y_train = data[1:, 4]

model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_split=0.2
)

x_predict = np.array([
  [10, 25, 14, 9],
  [102, 100, 75, 90]
])

# instead of `predict` us `predict_classes` to round the output
output = model.predict_classes(x_predict)

print("==output==")
print(output)