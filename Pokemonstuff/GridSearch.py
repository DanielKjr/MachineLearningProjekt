from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor



# MASTER INPUTS
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset') # Database Directory
epochCount = 2
batch_size = 32
img_height = 180
img_width = 180
# MASTER INPUTS



train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#lav det om til noget gridsearh kan bruge
X_train = []
y_train = []
for images, labels in train_ds:
    X_train.append(images.numpy())
    y_train.append(labels.numpy())

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

X_val = []
y_val = []
for images, labels in val_ds:
    X_val.append(images.numpy())
    y_val.append(labels.numpy())

X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)
def buildModel(optimizer):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=128, activation='relu'))

    # Adding the second hidden layer
    classifier.add(Dense(units=64, activation='relu'))

    classifier.add(Dense(units=32, activation='relu'))

    # Adding the output layer
    classifier.add(Dense(units=1, activation='linear'))

    # Compiling the ANN
    classifier.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    return classifier

classifier = KerasRegressor(build_fn = buildModel)
#What hyperparameter we want to play with
parameters = {'batch_size': [16, 32, 64, 128],
              'epochs': [3, 7],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 5)
grid_search.fit(X_train, y_train, validation_data=(X_val, y_val))
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: " + str(best_parameters))