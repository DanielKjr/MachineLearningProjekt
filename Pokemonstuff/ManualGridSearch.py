import os
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
num_epochs = [4, 8]
optimizers = ['adam', 'sgd', 'rmsprop']
data_dir = os.path.expanduser(fr'~\Desktop\Pokemon\Pokemon Dataset\Pokemon Dataset')
num_classes = 973
best_accuracy = 0
best_hyperparameters = {}

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
                tf.keras.layers.Dense(num_classes, activation='softmax')
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

            # clone model to get a fresh version each test
            model_clone = tf.keras.models.clone_model(model)

            model_clone.compile(
                optimizer=optimizer_instance,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )


            for epoch_count in num_epochs:
                print(f"Training model with lr={lr}, batch_size={bs}, epochs={epoch_count}, optimizer={optimizer}")

                history = model_clone.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epoch_count,
                    batch_size=bs
                )


                accuracy = history.history['val_accuracy'][0]

                # Update best hyperparameters
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = {'learning_rate': lr, 'batch_size': bs, 'epochs': epoch_count,
                                            'optimizer': optimizer}

                # Plot accuracy and loss
                axs[i, j].plot(history.history['accuracy'], label=f'accuracy (epochs={epoch_count})')
                axs[i, j].plot(history.history['val_accuracy'], label=f'val_accuracy (epochs={epoch_count})')
                axs[i, j].plot(history.history['loss'], label=f'loss (epochs={epoch_count})')
                axs[i, j].plot(history.history['val_loss'], label=f'val_loss (epochs={epoch_count})')
                axs[i, j].set_title(f'lr={lr}, batch_size={bs}')
                axs[i, j].set_xlabel('Epoch')
                axs[i, j].set_ylabel('Metric')
                if j == 0 and i == 0:
                    legend = axs[i, j].legend(loc='upper left')
    
    # Add the figure to the list of all figures
    all_figures.append(fig)
    try:
        plt.savefig(f'plot-{optimizer}.png')
    except Exception as e:
        print(f"Erro when saving f'plot-{optimizer}.png'", e)

print("Best hyperparameters:", best_hyperparameters)
print("Best accuracy:", best_accuracy)

try:
    with open("best_params.txt", "w") as file:
        file.write("Best hyperparameters: {}\n".format(best_hyperparameters))
        file.write("Best accuracy: {}\n".format(best_accuracy))
except Exception as e:
    print("Error when saving best_params.txt", e)


# Show all figures together
plt.show()