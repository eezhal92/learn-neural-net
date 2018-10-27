"""
Train to predict mean
Classification
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
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

# learning rate is how large the steps are that the network takes while learning
# If it's too small, the network will never have a chance to get where it's going
# And accuracy will be always low, or training will be take a really long time
# But if it's too large the network will jump all over the place and never be able 
# to find the best solution because it wil keep jumping over
# It's important to change the learning rate for the network and test it
# because each datasets in network topology will respond slightly differently to different
# learning rates
opt = Adam(lr=0.005) # default 0.001

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

# we can change learning rate after a fit, with lower learning rate to improve the neural network
model.optimizer = Adam(lr=0.0001)

model.fit(
  x_train,
  y_train,
  epochs=100,
  batch_size=2,
  verbose=1,
  validation_split=0.2
)

model.save('iris.h5')

"""
What we're doing by first writing with a high learning rate and then switching to a small learning rate is telling the network that it can start by taking large steps, which gives it more freedom to explore the training landscape.
Then when we want to start refining the results, without risking taking a big step in the wrong direction, we lower the learning rate and continue training.
Now when we run that it starts with a hundred epochs at the first learning rate and then continues with another hundred epochs at the smaller learning rate.
Now that we're happy with that model, let's save it, so that we can reload the fully-trained model later.
"""