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
import jsonpickle
import numpy as np
import pandas as pd
from keras.callbacks import History
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, load_model

# from keras.optimizers.adam import Adam
from keras.optimizers.adamw import AdamW
from keras.utils import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tensorflow import lite

from modules.config import (
    BATCH_SIZE,
    IMG_SIZE,
    INIT_LR,
    NUM_CLASSES,
    NUM_EPOCHS,
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
    def __init__(self, interpreter):
        self.interpreter: lite.Interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[1]
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


class CustomModel:
    def __init__(
        self, base_model_function: Model, trained: bool, lite_model_required: bool
    ) -> None:
        logger.info(f"\t$BLUEstarting `custom_model_instance.__init__`...$RESET")
        start = timer()
        self.model: Model = None
        self.history: dict = None
        self.lite_model_required = lite_model_required
        self.tflite_model = None
        input_shape = IMG_SIZE + tuple([3])
        input_tensor = Input(shape=IMG_SIZE + tuple([3]))
        base_model_args = dict(
            input_shape=input_shape,
            weights="imagenet",
            include_top=False,
            input_tensor=input_tensor,
        )
        self.base_model: Model = base_model_function(**base_model_args)
        logger.info(
            f"\t\t$BLUE`custom_model_instance.base_model.__init__` took $RED{timer()-start:.2f} $BLUEseconds$RESET"
        )
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
            self.define_model()
            self.history = self.train().history
        else:
            self.model = load_model(self.saved_model_path)
            with open(self.history_path, "r") as f:
                self.history = jsonpickle.decode(f.read())
        logger.info(
            f"\t\t $BLUE`custom_model_instance.model` took $RED{timer()-start:.2f} $BLUEseconds$RESET"
        )
        if self.lite_model_required:
            lite_model_path = pathlib.Path(self.saved_model_path).with_suffix(".tflite")
            if os.path.exists(lite_model_path):
                with open(lite_model_path, "rb") as f:
                    self.tflite_model = f.read()
                    self.tflite_model_instance = LiteModel(
                        lite.Interpreter(model_path=str(lite_model_path))
                    )
            else:
                try:
                    self.tflite_model = lite.TFLiteConverter.from_keras_model(
                        self.model
                    ).convert()
                    self.tflite_model_instance = LiteModel(
                        lite.Interpreter(model_content=self.tflite_model)
                    )
                    with open(lite_model_path, "wb") as f:
                        f.write(self.tflite_model)
                except Exception:
                    logger.error(
                        f"$REDCould not convert model to `tf.lite` model$RESET",
                        exc_info=1,
                    )
                    exit(0)
        logger.info(
            f"\t$BLUE`custom_model_instance.__init__` took $RED{timer()-start:.2f} $BLUEseconds$RESET"
        )

    def train(self) -> History:
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

        logger.info(f"\t$BLUEstarting training...$RESET")
        start = timer()
        # Train the model
        history = self.model.fit(
            x_train,
            train_targets,
            validation_data=(x_validation, validation_targets),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        logger.info(f"\t$BLUEending training...$RESET")
        logger.info(f"\t$BLUEtraining took $RED{timer()-start:.2f} $BLUEseconds$RESET")
        self.model.save(self.saved_model_path)
        with open(self.history_path, "w") as f:
            f.write(jsonpickle.encode(history.history))
        return history

    def test_model_images(self, only_random: bool = False):
        start = timer()
        images, bboxes, image_paths = load_test_data(test_data_dir)
        if os.path.exists(self.predicted_labels_path):
            with open(self.predicted_labels_path, "rb") as f:
                predicted_labels = pickle.load(f)
        else:
            logger.info(f"\t$BLUEpredicting labels for images...$RESET")
            predicted_labels = self.model.predict(
                images,
                batch_size=BATCH_SIZE,
                verbose=0,
            )[1]
            logger.info(
                f"\t$BLUEpredicting labels took $RED{timer()-start:.2f} $BLUEseconds$RESET"
            )
            with open(self.predicted_labels_path, "wb") as f:
                pickle.dump(predicted_labels, f)
        predicted_labels = np.array(predicted_labels)
        with open(os.path.join(test_data_dir, "Test.csv")) as f:
            correct_labels = pd.read_csv(f, sep=",")["ClassId"].to_numpy(dtype="uint32")
        with open(labels_path, "r") as f:
            labels_json = json.load(f)

        if only_random:
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
                logger.info(f"\t$BLUEtesting random images took $RED{timer()-start:.2f} $BLUEseconds$RESET")
        else:
            if not os.path.exists(self.scores_path):
                logger.info(
                    f"\t$BLUEevaluating model $RED{self.base_model.name} $BLUEand saving scores...$RESET"
                )
                testTargets = {"class_label": predicted_labels, "bounding_box": bboxes}
                metrics_names: list[str] = self.model.metrics_names

                scores = self.model.evaluate(
                    images,
                    testTargets,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )
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
                correct = 0
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
                with open(self.scores_path, "a") as f:
                    f.write(test_acc)
                logger.info(f"\t$BLUEtesting all images took $RED{timer()-start:.2f} $BLUEseconds$RESET")

    def test_model_videos(self, input_video_fn: str):
        logger.info(
            f"$BOLD$COLOR==> $BOLD$BLUEprocessing input video $GREEN{input_video_fn}$RESET"
        )
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
        video_stream = ffmpeg.probe(input_video_path)["streams"][0]
        ns = {"__builtins__": None}
        # frame_height = int(video_stream["height"])
        # frame_width = int(video_stream["width"])
        fps = math.ceil(float(eval(video_stream["avg_frame_rate"], ns)))
        # pix_fmt = video_stream["pix_fmt"]
        pix_fmt = "rgb24"
        ffmpeg_args = {
            "hide_banner": None,
            "loglevel": "quiet",
            "v": "quiet",
            "nostats": None,
        }
        if not os.path.exists(pathlib.Path(input_frames_path).parent):
            os.mkdir(pathlib.Path(input_frames_path).parent)
        if not os.path.exists(pathlib.Path(output_video_path).parent):
            os.mkdir(pathlib.Path(output_video_path).parent)
        generated_frames = os.path.exists(input_frames_path)
        generated_video = os.path.exists(output_video_path)
        if generated_frames:
            if generated_video:
                logger.info(
                    f"\t$BLUEalready generated frames and video for video $GREEN{input_video_fn}$RESET"
                )
            else:
                logger.info(
                    f"\t$BLUEreading generated frames for video $GREEN{input_video_fn}$RESET"
                )
                with gzip.GzipFile(input_frames_path, "r") as f:
                    resized_frames = np.load(f)
                logger.info(f"writing output video $GREEN{output_video_path}$RESET")
                vidwrite(
                    output_video_path,
                    resized_frames,
                    fps=fps // 4,
                    in_pix_fmt=pix_fmt,
                    input_args=ffmpeg_args,
                    output_args={
                        i: ffmpeg_args[i] for i in ffmpeg_args if i != "hide_banner"
                    },
                )
                return
        else:
            logger.info(f"\t$BLUEgenerating frames")
            vidcap = cv2.VideoCapture(input_video_path)
            resized_frames = []
            frames = []
            count = 0
            while vidcap.isOpened():
                success, frame = vidcap.read()
                if success:
                    img = Image.fromarray(frame)
                    frames.append(img)
                    resized_frame = cv2.resize(frame, (64, 64))
                    resized_frames.append(resized_frame)
                    count += fps
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    vidcap.release()
                    break
            resized_frames = np.array(resized_frames)
            start = timer()
            if not self.lite_model_required:
                label_predictions = self.model.predict(
                    resized_frames, verbose=0, batch_size=BATCH_SIZE
                )
            else:
                label_predictions = self.tflite_model_instance.predict(resized_frames)
            logger.info(
                f"\t$BLUEpredicting labels took $RED{timer()-start:.2f} $BLUEseconds$RESET"
            )
            start = timer()
            for index, frame in zip(range(resized_frames.shape[0]), frames):
                label = labels_json[str(np.argmax(label_predictions[index]))]
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
                frame.paste(button_img, (0, 0))
                frames[index] = np.array(frame, dtype=np.uint8)
            logger.info(
                f"\t$BLUEmodifying images took $RED{timer()-start:.2f} $BLUEseconds$RESET"
            )
            logger.info(
                f"\t$BLUEsaving $RED{len(resized_frames)}$BLUE  frames in $GREEN{input_frames_path}$RESET"
            )
            with gzip.GzipFile(input_frames_path, mode="w", compresslevel=3) as f:
                np.save(f, resized_frames)
            logger.info(f"\t$BLUEwriting output video $GREEN{output_video_path}$RESET")
            vidwrite(
                output_video_path,
                resized_frames,
                fps=fps // 4,
                in_pix_fmt=pix_fmt,
                input_args=ffmpeg_args,
                output_args={
                    i: ffmpeg_args[i] for i in ffmpeg_args if i != "hide_banner"
                },
            )
            return

    def define_model(self):
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
        if self.lite_model_required == False:
            softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        if self.lite_model_required == False:
            softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512, activation="relu")(softmaxHead)
        if self.lite_model_required == False:
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
        opt = AdamW(INIT_LR)
        self.model.compile(
            loss=losses,
            optimizer=opt,
            metrics=["accuracy"],
            loss_weights=lossWeights,
        )
