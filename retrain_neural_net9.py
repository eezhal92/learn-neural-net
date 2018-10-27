
"""
Continue retrain saved model
 This can be especially helpful if you get new data and you want to train an old model, 
 or if you simply want to pause training because it's taking a long time and resume at a later time.
"""

import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical

model = load_model('iris.h5')
model.summary()

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
