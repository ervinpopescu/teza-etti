import os

from keras.models import Model
from keras.utils.vis_utils import plot_model
from modules.config import BATCH_SIZE, NUM_EPOCHS, saved_model_path, training_data_dir
from modules.load_data import load_training_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def train_model(model: Model) -> None:
    # Load the data
    images, labels, bboxes, _ = load_training_data(training_data_dir)
    split = train_test_split(images, labels, bboxes, test_size=0.2, random_state=12)

    (x_train, x_validation) = split[0:2]
    (y_train, y_validation) = split[2:4]
    (bboxes_train, bboxes_validation) = split[4:6]

    train_targets = {"class_label": y_train, "bounding_box": bboxes_train}
    validation_targets = {
        "class_label": y_validation,
        "bounding_box": bboxes_validation,
    }

    model.summary()
    if not os.path.exists("../../figuri/model_plot.png"):
        plot_model(
            model,
            to_file="../../figuri/model_plot.png",
            dpi=192,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            show_trainable=True,
        )

    # Train the model
    # history ls= model.fit(
    model.fit(
        x_train,
        train_targets,
        validation_data=(x_validation, validation_targets),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    model.save(saved_model_path)

    # Plot the training and validation accuracy over time
    # plt.plot(history.history["class_label_accuracy"], label="accuracy")
    # plt.plot(history.history["val_accuracy"], label="val_accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="lower right")
    # plt.show()

    return
