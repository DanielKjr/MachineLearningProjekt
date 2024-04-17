import numpy as np
import os
import tensorflow as tf
from keras.optimizers import RMSprop

data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset')  # Database Directory
model_input = "pokemon.keras"
model_output = "pokemon_1.keras"
epochCount = 8
loadExistingModel = False
batch_size = 16
img_height = 180
img_width = 180
learnRate = 0.001

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

class_names = train_ds.predictionClasses
num_classes = 973  # Amount of pokemons in dataset

#compile kun modellen hvis den ikke loades
if loadExistingModel:
    model = tf.keras.models.load_model(model_input)
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_height,img_width, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=RMSprop(learnRate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochCount
)

model.save(model_output)
