import os

import jsonpickle
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.optimizers.rmsprop import RMSprop
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

from modules.config import (BATCH_SIZE, IMG_SIZE, INIT_LR, NUM_CLASSES,
                            NUM_EPOCHS, output_path, training_data_dir)
from modules.load_data import load_training_data


class CustomModel:
    def __init__(self, base_model_function: Model) -> None:
        self.model: Model = None
        self.base_model: Model = base_model_function(
            input_shape=IMG_SIZE + tuple([3]),
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=IMG_SIZE + tuple([3])),
        )
        # freeze training any of the layers of the base model
        for layer in self.base_model.layers:
            layer.trainable = False

        flatten = self.base_model.output
        flatten = Flatten()(flatten)

        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
        # 4 neurons correspond to 4 co-ords in output bbox

        softmaxHead = Dense(512, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(NUM_CLASSES, activation="softmax", name="class_label")(
            softmaxHead
        )
        self.model = Model(
            inputs=self.base_model.input, outputs=(bboxHead, softmaxHead)
        )

        losses = {
            "class_label": "categorical_crossentropy",
            "bounding_box": "mean_squared_error",
        }
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = RMSprop(INIT_LR)
        self.model.compile(
            loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights
        )

        self.saved_model_path = os.path.join(
            output_path, self.base_model.name, "model.h5"
        )
        self.history_path = os.path.join(
            output_path, self.base_model.name, "training_history.jsonpickle"
        )
        self.scores_path = os.path.join(output_path, self.base_model.name, "scores.txt")
        self.lb_path = os.path.join(output_path, self.base_model.name, "lb.pickle")
        self.predicted_labels_path = os.path.join(
            output_path, self.base_model.name, "predicted_labels.pickle"
        )
        self.accuracies_path = os.path.join(
            output_path, self.base_model.name, "accuracies.txt"
        )

    def train(self):
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

        self.model.summary()

        if not os.path.exists(f"../../figuri/{self.base_model.name}/model_plot.png"):
            plot_model(
                self.model,
                to_file=f"../../figuri/{self.base_model.name}/model_plot.png",
                dpi=192,
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=True,
                show_trainable=True,
            )

        # Train the model
        history = self.model.fit(
            x_train,
            train_targets,
            validation_data=(x_validation, validation_targets),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        self.model.save(self.saved_model_path)

        with open(self.history_path, "w") as f:
            f.write(jsonpickle.encode(history))

        return history
