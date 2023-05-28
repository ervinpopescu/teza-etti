import os
import pathlib

from keras.applications import Xception
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.optimizers.rmsprop import RMSprop

current_file_path = pathlib.Path(__file__).parent.parent
input_path = os.path.join(current_file_path, "input")
output_path = os.path.join(current_file_path, "output")


# Define the location of the dataset
training_data_dir = os.path.join(input_path, "images", "Training")
test_data_dir = os.path.join(input_path, "images", "Test")
input_videos_dir = os.path.join(input_path, "videos")

# Define the image size and number of classes
IMG_SIZE = (64, 64)
VIDEO_SIZE = (1024, 1024)
NUM_CLASSES = 43
INIT_LR = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 64
labels_path = os.path.join(input_path, "labels.json")
input_videos_filenames = os.listdir(input_videos_dir)
lb_path = os.path.join(output_path, "lb.pickle")
predicted_labels_path = os.path.join(output_path, "predicted_labels.pickle")
saved_model_path = os.path.join(output_path, "model.h5")
scores_path = os.path.join(output_path, "scores.txt")


def define_model() -> Model:
    # Define the model
    vgg = Xception(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=IMG_SIZE + tuple([3])),
    )

    # freeze training any of the layers of VGGNet
    for layer in vgg.layers:
        layer.trainable = False

    # vgg.summary()

    # max-pooling is output of VGG, flattening it further
    flatten = vgg.output
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
    model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))

    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    }
    lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
    opt = RMSprop(INIT_LR)
    model.compile(
        loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights
    )

    return model
