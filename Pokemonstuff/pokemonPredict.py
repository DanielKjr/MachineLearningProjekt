import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras



# MASTER INPUTS
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset') # Database Directory
model_name = "pokemon.keras"
# MASTER INPUTS



batch_size = 32
img_height = 180
img_width = 180

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

class_names = train_ds.class_names
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = 973 # Amount of pokemons in dataset

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
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
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model = tf.keras.models.load_model(model_name) # LOAD THE SAVED MODEL ---------------------------------------------------------------------- LOAD THE SAVED MODEL

for images, labels in val_ds:
  # Make predictions using your model
  predictions = model.predict(images)

  # Visualize the images along with their predicted labels
  for i in range(len(images)):
    image = images[i]
    label = labels[i]
    predicted_label = np.argmax(predictions[i])

    # Convert label indices to class names
    true_class_name = class_names[label]
    predicted_class_name = class_names[predicted_label]

    # Display the image along with true and predicted labels
    plt.figure()
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(f"True: {true_class_name}, Predicted: {predicted_class_name}")
    plt.axis("off")
    plt.show()
