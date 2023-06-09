import gzip
import json
import math
import os
import pathlib
import pickle
import random
from timeit import default_timer as timer

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers.rmsprop import RMSprop
from keras.utils import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tensorflow import lite

from modules.config import (
    BATCH_SIZE,
    BLUE,
    GREEN,
    IMG_SIZE,
    INIT_LR,
    NUM_CLASSES,
    NUM_EPOCHS,
    RED,
    RESET,
    input_path,
    input_videos_dir,
    labels_path,
    logger,
    output_path,
    test_data_dir,
    training_data_dir,
)
from modules.load_data import load_test_data, load_training_data
from modules.videowriter import vidwrite


class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter: lite.Interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()
        output_det = output_det[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp: np.ndarray):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i : i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """Like predict(), but only for a single record. The input data can be a Python list."""
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


class CustomModel:
    def __init__(self, base_model_function: Model, trained: bool) -> None:
        self.model: Model = None
        input_shape = IMG_SIZE + tuple([3])
        input_tensor = Input(shape=IMG_SIZE + tuple([3]))
        base_model_args = dict(
            input_shape=input_shape,
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,
        )
        self.base_model: Model = base_model_function(**base_model_args)
        self.saved_model_path = os.path.join(
            output_path, self.base_model.name, "model.h5"
        )
        self.history_path = os.path.join(
            output_path, self.base_model.name, "training_history.json"
        )
        self.scores_path = os.path.join(output_path, self.base_model.name, "scores.txt")
        self.lb_path = os.path.join(output_path, self.base_model.name, "lb.pickle")
        self.predicted_labels_path = os.path.join(
            output_path, self.base_model.name, "predicted_labels.pickle"
        )
        self.accuracies_path = os.path.join(
            output_path, self.base_model.name, "accuracies.txt"
        )
        if not trained:
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
                loss=losses,
                optimizer=opt,
                metrics=["accuracy"],
                loss_weights=lossWeights,
            )
        else:
            self.model = load_model(self.saved_model_path)

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

        # self.model.summary()

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

        logger.info(f"{BLUE}starting training...{RESET}")
        # Train the model
        history = self.model.fit(
            x_train,
            train_targets,
            validation_data=(x_validation, validation_targets),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        logger.info(f"{BLUE}ending training...{RESET}")
        self.model.save(self.saved_model_path)

        return history

    def test_model_images(self, include_random: bool = False):
        images, bboxes, image_paths = load_test_data(test_data_dir)
        if os.path.exists(self.predicted_labels_path):
            with open(self.predicted_labels_path, "rb") as f:
                predicted_labels = pickle.load(f)
        else:
            logger.info(f"{BLUE}predicting labels for images...{RESET}")
            predicted_labels = self.model.predict(
                images,
                batch_size=BATCH_SIZE,
                verbose=0,
            )[1]
            with open(self.predicted_labels_path, "wb") as f:
                pickle.dump(predicted_labels, f)
        predicted_labels = np.array(predicted_labels)
        with open(os.path.join(test_data_dir, "Test.csv")) as f:
            correct_labels = pd.read_csv(f, sep=",")["ClassId"].to_numpy(dtype="uint32")
        with open(labels_path, "r") as f:
            labels_json = json.load(f)

        testTargets = {"class_label": predicted_labels, "bounding_box": bboxes}
        metrics_names: list[str] = self.model.metrics_names
        correct = 0

        if not os.path.exists(self.scores_path):
            logger.info(
                f"{BLUE}evaluating model {GREEN}{self.base_model.name} {BLUE}and saving scores...{RESET}"
            )
            scores = self.model.evaluate(
                images,
                testTargets,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

            for image_path in image_paths:
                index = np.where(image_paths == image_path)[0][0]
                image = load_img(image_path, target_size=IMG_SIZE)
                image = img_to_array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                # # finding class label with highest pred. probability
                i = np.argmax(predicted_labels[index], axis=0)
                predicted_label = labels_json[str(i)]
                correct_label = labels_json[str(correct_labels[index])]

                if predicted_label == correct_label:
                    correct += 1

            test_acc = f"Test accuracy: {correct/len(images)*100:.2f}%\n"
            with open(self.scores_path, "w") as f:
                for name, score in zip(metrics_names, scores):
                    name = name.split("_")
                    name[0] = name[0].capitalize()
                    joined_name = " ".join(name)
                    if "Loss" in joined_name or "loss" in joined_name:
                        line = "{}: {:.2f}\n".format(joined_name, score)
                    else:
                        line = "{}: {:.2f}%\n".format(joined_name, score * 100)
                    f.write(line)
                f.write(test_acc)
        if include_random:
            logger.info(f"")
            for i in range(100):
                correct = 0
                random.seed(random.random() * 50)
                random_choices = random.choices(
                    image_paths, k=int(len(image_paths) / 100)
                )
                for image_path in random_choices:
                    index = np.where(image_paths == image_path)[0][0]
                    i = np.argmax(predicted_labels[index], axis=0)
                    predicted_label = labels_json[str(i)]
                    correct_label = labels_json[str(correct_labels[index])]
                    if predicted_label == correct_label:
                        correct += 1
                random_acc = f"{correct/len(random_choices)*100:.2f}\n"
                with open(self.accuracies_path, "a") as f:
                    f.write(random_acc)

    def test_model_videos(self, input_video_fn: str):
        logger.info(f"processing input video {GREEN}{input_video_fn}{RESET}")
        input_video_path = os.path.join(input_videos_dir, input_video_fn)
        output_video_path = os.path.join(
            output_path,
            self.base_model.name,
            f'output-{input_video_fn.replace(".mp4","")}.mp4',
        )
        input_frames_path = os.path.join(
            input_path,
            "frames",
            self.base_model.name,
            f'{input_video_fn.replace(".mp4","")}.frames.npy.gz',
        )
        with open(labels_path, "r") as f:
            labels_json = json.load(f)
        lite_model_path = pathlib.Path(self.saved_model_path).with_suffix(".tflite")
        if os.path.exists(lite_model_path):
            with open(lite_model_path, "rb") as f:
                model = LiteModel.from_file(model_path=str(lite_model_path))
        else:
            try:
                model = LiteModel.from_keras_model(self.model)
                with open(lite_model_path, "wb") as f:
                    f.write(model)
            except:
                model = self.model
        video_stream = ffmpeg.probe(input_video_path)["streams"][0]
        ns = {"__builtins__": None}
        # frame_height = int(video_stream["height"])
        # frame_width = int(video_stream["width"])
        fps = math.ceil(float(eval(video_stream["avg_frame_rate"], ns)))
        # pix_fmt = video_stream["pix_fmt"]
        pix_fmt = "rgb24"
        if not os.path.exists(pathlib.Path(input_frames_path).parent):
            os.mkdir(pathlib.Path(input_frames_path).parent)
        if not os.path.exists(pathlib.Path(output_video_path).parent):
            os.mkdir(pathlib.Path(output_video_path).parent)
        generated_frames = os.path.exists(input_frames_path)
        generated_video = os.path.exists(output_video_path)
        if generated_frames:
            if generated_video:
                logger.info(
                    f"already generated frames and video for video {GREEN}{input_video_fn}{RESET}"
                )
            else:
                logger.info(
                    f"reading generated frames for video {GREEN}{input_video_fn}{RESET}"
                )
                with gzip.GzipFile(input_frames_path, "r") as f:
                    frames = np.load(f)
                logger.info(f"writing output video {GREEN}{output_video_path}{RESET}")
                vidwrite(
                    output_video_path,
                    frames,
                    fps=fps // 4,
                    in_pix_fmt=pix_fmt,
                    input_args={
                        "hide_banner": None,
                        "loglevel": "quiet",
                        "nostats": None,
                    },
                    output_args={
                        "loglevel": "quiet",
                        "v": "quiet",
                        "nostats": None,
                    },
                )
                return
        else:
            logger.info(f"generating frames for video {GREEN}{input_video_fn}{RESET}")
            vidcap = cv2.VideoCapture(input_video_path)
            frames = []
            count = 0
            while vidcap.isOpened():
                success, frame = vidcap.read()
                if success:
                    image = Image.fromarray(frame)
                    frame = cv2.resize(frame, (64, 64))  # resize the frame
                    expanded_frame = np.expand_dims(
                        frame, axis=0
                    )  # add an extra dimension to make it a batch of size 1
                    if count == 0:
                        start = timer()
                    if isinstance(model, Model):
                        label_predictions = model.predict(expanded_frame, verbose=0)
                    elif isinstance(model, LiteModel):
                        label_predictions = model.predict(expanded_frame)
                    if count == 0:
                        logger.info(
                            f"predicting label took {RED}{timer()-start:.6f} {BLUE}seconds{RESET}"
                        )
                    label = labels_json[str(np.argmax(label_predictions))]
                    # logger.info(label)
                    if count == 0:
                        start = timer()
                    font = ImageFont.truetype(
                        "/usr/share/fonts/OTF/intelone-mono-font-family-regular.otf",
                        size=20,
                    )
                    margin = 10
                    left, top, right, bottom = font.getbbox(label)
                    width, height = right - left, bottom - top
                    button_size = (width + 2 * margin, height + 3 * margin)
                    button_img = Image.new("RGBA", button_size, "black")
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text((10, 10), label, fill=(0, 255, 0), font=font)
                    image.paste(button_img, (0, 0))
                    frames.append(np.array(image, dtype=np.uint8))
                    if count == 0:
                        logger.info(
                            f"modifying image took {RED}{timer()-start:.6f} {BLUE}seconds{RESET}"
                        )
                    count += fps  # i.e. at 30 fps, this advances one second
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    vidcap.release()
                    break
            frames = np.array(frames)
            logger.info(
                f"saving {RED}{len(frames)}{BLUE} frames in {GREEN}{input_frames_path}{RESET}"
            )
            with gzip.GzipFile(input_frames_path, mode="w", compresslevel=3) as f:
                np.save(f, frames)
            logger.info(f"writing output video {GREEN}{output_video_path}{RESET}")
            vidwrite(
                output_video_path,
                frames,
                fps=fps // 4,
                in_pix_fmt=pix_fmt,
                input_args={
                    "hide_banner": None,
                    "loglevel": "quiet",
                    "nostats": None,
                    "v": "quiet",
                },
                output_args={"loglevel": "quiet", "nostats": None, "v": "quiet"},
            )
            return
