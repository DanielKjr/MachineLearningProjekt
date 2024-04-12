import os
import tensorflow as tf


# Define hyperparameters to experiment with
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
# batch_sizes = [1]
num_epochs = [3, 5, 8]
# num_epochs = [1]
optimizers = ['adam', 'sgd', 'rmsprop']

best_accuracy = 0
best_hyperparameters = {}

# Load and preprocess data
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset')
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

for optimizer in optimizers:

    # Perform manual hyperparameter search
    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in num_epochs:
                print(f"Training model with lr={lr}, batch_size={bs}, epochs={epochs}, optimizer={optimizer}")

                # Define and compile the model
                model = tf.keras.Sequential([
                    tf.keras.layers.Rescaling(1. / 255),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(973, activation='softmax')
                ])

                if optimizer == 'adam':
                    optimizer_instance = tf.keras.optimizers.Adam(learning_rate=lr)
                elif optimizer == 'sgd':
                    optimizer_instance = tf.keras.optimizers.SGD(learning_rate=lr)
                elif optimizer == 'rmsprop':
                    optimizer_instance = tf.keras.optimizers.RMSprop(learning_rate=lr)
                else:
                    raise ValueError(f"Unknown optimizer: {optimizer}")

                model.compile(
                    optimizer=optimizer_instance,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

                # Train the model
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    batch_size=bs,
                    verbose=0
                )

                # Evaluate model performance
                _, accuracy = model.evaluate(val_ds, verbose=0)

                # Update best hyperparameters if necessary
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = {'learning_rate': lr, 'batch_size': bs, 'epochs': epochs,
                                            'optimizer': optimizer}



print("Best hyperparameters:", best_hyperparameters)
print("Best accuracy:", best_accuracy)
