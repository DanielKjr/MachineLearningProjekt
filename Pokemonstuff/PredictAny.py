import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


model_input = "pokemon.keras"
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset')
img_height = 180
img_width = 180
batch_size = 32
# Load the trained model




train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # validation_split=0.2,
  # subset="training",
  # seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
model = tf.keras.models.load_model(model_input)

# Preprocess the image
def preprocess_image(image_path):
    # img = Image.open(image_path)
    # img = img.resize((img_height, img_width))
    #
    # # Ensure image has three channels (RGB)
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    #
    # img_array = np.array(img) / 255.0  # Normalize pixel values
    # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img = mpimg.imread(image_path)
    if img.shape[2] == 4:
        # Convert RGBA to RGB by discarding the alpha channel
        img = img[:, :, :3]

    # Resize the image
    img = tf.image.resize(img, [img_height, img_width])

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)
    return img_array

# Make predictions
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence

# Example usage
image_path = r"C:\Users\danie\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset\Pikachu\Generation 2\Gold\Back\Pikachu_1_2.png"
# predicted_class, confidence = predict_image(image_path)
# print("Predicted Pokemon:", predicted_class)
# print("Confidence:", confidence)
#
# img = Image.open(image_path)
#
# Display the image with the predicted label
# plt.imshow(img)
# plt.title(f"Predicted Pokemon: {predicted_class}")
# plt.axis('off')
# plt.show()
# Load the single image
# image_path = r"C:\Users\danie\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset\Pikachu\Generation 2\Gold\Back\Pikachu_1_2.png"
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))

# Convert the image to a numpy array
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# Preprocess the image (same preprocessing as in the dataset)
img_array = img_array / 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(img_array)

# Get predicted label
predicted_label_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_label_index]

print("Predicted Pokemon:", predicted_class_name)
