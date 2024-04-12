import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define hyperparameters to experiment with
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
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

all_figures = []

for optimizer in optimizers:

    # Create a subplot for each optimizer
    fig, axs = plt.subplots(len(learning_rates), len(batch_sizes), figsize=(15, 15))
    fig.suptitle(f'Optimizer: {optimizer}')

    for i, lr in enumerate(learning_rates):
        for j, bs in enumerate(batch_sizes):

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

            # Perform experiments for each num_epochs
            for epoch_count in num_epochs:
                print(f"Training model with lr={lr}, batch_size={bs}, epochs={epoch_count}, optimizer={optimizer}")

                # Clone the model to ensure a fresh start for each experiment
                model_clone = tf.keras.models.clone_model(model)
                model_clone.compile(
                    optimizer=optimizer_instance,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

                # Train the model
                history = model_clone.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epoch_count,
                    batch_size=bs,
                    verbose=0
                )

                # Plot accuracy and loss
                axs[i, j].plot(history.history['accuracy'], label=f'accuracy (epochs={epoch_count})')
                axs[i, j].plot(history.history['val_accuracy'], label=f'val_accuracy (epochs={epoch_count})')
                axs[i, j].plot(history.history['loss'], label=f'loss (epochs={epoch_count})')
                axs[i, j].plot(history.history['val_loss'], label=f'val_loss (epochs={epoch_count})')
                axs[i, j].set_title(f'lr={lr}, batch_size={bs}')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Metric')
                axs[i, j].legend()

    # Add the figure to the list of all figures
    all_figures.append(fig)

# Show all figures together

# Show all figures together
plt.show()

print("Best hyperparameters:", best_hyperparameters)
print("Best accuracy:", best_accuracy)
