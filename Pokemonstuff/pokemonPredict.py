import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras



# MASTER INPUTS
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset') # Database Directory
model_input = "RMSProp.keras"
# MASTER INPUTS



batch_size = 32
img_height = 180
img_width = 180

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=12,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_ds.class_names

num_classes = 973 # Amount of pokemons in dataset
model = tf.keras.models.load_model(model_input) # LOAD THE SAVED MODEL ---------------------------------------------------------------------- LOAD THE SAVED MODEL

for images, labels in val_ds:
  predictions = model.predict(images)

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
