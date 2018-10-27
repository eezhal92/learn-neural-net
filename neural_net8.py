"""
Train to predict mean
Classification
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
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
model.add(Dense(3, activation='softmax')) # softmax ensure that all three probabilities add to one. help neural net to decide which class it might belong to

# 0 = low
# 1 = high

model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', # optimize data for being a part of multiple classes
  metrics=['accuracy']
)

data = np.genfromtxt('iris.csv', delimiter=',')
x_train = data[1:, :4]
y_train = to_categorical(data[1:, 4])

# shuffle data
perm = np.random.permutation(y_train.shape[0])
x_train = x_train[perm]
y_train = y_train[perm]


model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_split=0.2
)

x_predict = np.array([
  [4.9, 3.0, 1.5, 0.2], # 0
  [5.7, 3.0, 4.5, 1.2], # 1
  [7.2, 3.2, 6.4, 2.3], # 2
])

# See the probabilities
output = model.predict(x_predict)
# To make it easier to read
np.set_printoptions(suppress=True)

print("==output==")
print(output)