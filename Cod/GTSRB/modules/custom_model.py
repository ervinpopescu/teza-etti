import gzip
import json
import math
import os
import pickle
import random
import time

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers.rmsprop import RMSprop
from keras.utils import img_to_array, load_img
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from modules.config import (BATCH_SIZE, BLUE, GREEN, IMG_SIZE, INIT_LR,
                            NUM_CLASSES, NUM_EPOCHS, RED, RESET, input_path,
                            input_videos_dir, labels_path, output_path,
                            test_data_dir, training_data_dir)
from modules.load_data import load_test_data, load_training_data
from modules.videowriter import vidwrite
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split


class CustomModel:
    def __init__(
        self, base_model_function: Model, trained: bool
    ) -> None:
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

        print(BLUE+"starting training"+RESET)
        # Train the model
        history = self.model.fit(
            x_train,
            train_targets,
            validation_data=(x_validation, validation_targets),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        self.model.save(self.saved_model_path)

        return history

    def test_model_images(self):
        images, bboxes, image_paths = load_test_data(test_data_dir)
        if os.path.exists(self.predicted_labels_path):
            with open(self.predicted_labels_path, "rb") as f:
                predicted_labels = pickle.load(f)
        else:
            print(BLUE+"predicting labels..."+RESET)
            predicted_labels = self.model.predict(
                images, batch_size=BATCH_SIZE, verbose=1
            )[1]
            with open(self.predicted_labels_path, "wb") as f:
                pickle.dump(predicted_labels, f)
        predicted_labels = np.array(predicted_labels)
        with open(os.path.join(test_data_dir, "Test.csv")) as f:
            correct_labels = pd.read_csv(f, sep=",")["ClassId"].to_numpy(dtype="uint32")
        with open(labels_path, "r") as f:
            labels_json = json.load(f)

        testTargets = {"class_label": predicted_labels, "bounding_box": bboxes}
        metrics_names = self.model.metrics_names
        correct = 0

        if not os.path.exists(self.scores_path):
            # Evaluate the model
            scores = self.model.evaluate(
                images,
                testTargets,
                batch_size=BATCH_SIZE,
                verbose=1,
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

            printed = ""
            test_acc = f"Test accuracy: {correct/len(images)*100:.2f}%\n"

            with open(self.scores_path, "w") as f:
                for name, score in zip(metrics_names, scores):
                    if "loss" in name:
                        line = "{}: {:.2f}\n".format(name, score)
                    else:
                        line = "{}: {:.2f}%\n".format(name, score * 100)
                    f.write(line)
                    printed = printed + line
                f.write(test_acc)
                printed = printed + test_acc
        else:
            with open(self.scores_path, "r") as f:
                printed = f.read()

        # for i in range(1):
        #     t1=time.time()
        #     correct = 0
        #     random.seed(random.random() * 50)
        #     random_choices = random.choices(image_paths, k=int(len(image_paths) / 100))
        #     for image_path in random_choices:
        #         index = np.where(image_paths == image_path)[0][0]
        #         i = np.argmax(predicted_labels[index], axis=0)
        #         predicted_label = labels_json[str(i)]
        #         correct_label = labels_json[str(correct_labels[index])]
        #         if predicted_label == correct_label:
        #             correct += 1

        #         image = Image.open(image_path)
        #         image = Image.Image.resize(image, size=(256, 256))

        #         # scaling pred. bbox coords according to image dims
        #         (xmin, ymin, xmax, ymax) = bboxes[index]
        #         (h, w) = (image.height, image.width)
        #         xmin = int(xmin * w)
        #         ymin = int(ymin * h)
        #         xmax = int(xmax * w)
        #         ymax = int(ymax * h)

        #         # drawing bbox and label on image
        #         draw = ImageDraw.ImageDraw(image, "RGBA")
        #         draw.font = ImageFont.truetype(
        #             "/usr/share/fonts/OTF/intelone-mono-font-family-regular.otf", size=13
        #         )
        #         draw.fontmode = "L"
        #         draw.text((xmin, (ymax - 10) / 2), predicted_label, (0, 255, 0))

        #         draw.rectangle(
        #             xy=(
        #                 (xmin, ymax),
        #                 (xmax, ymin),
        #             ),
        #             fill=(0, 0, 0, 0),
        #             outline=(0, 255, 0),
        #         )

        #         # showing the output image
        #         plt.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
        #         plt.show()

        #     random_acc = f"{correct/len(random_choices)*100:.2f}\n"
        #     with open(self.accuracies_path, "a") as f:
        #         f.write(random_acc)
        #     t2=time.time()
        # print(f"{RED}Seconds to compute random accuracy: {t2-t1}{RESET}")
        # printed = printed + random_acc
        # print(BLUE+printed+RESET)

        # return random_acc

    def test_model_videos(self, input_video_fn: str):
        print(f"{BLUE}processing input video {RED}{input_video_fn}{RESET}")
        input_video_path = os.path.join(input_videos_dir, input_video_fn)
        output_video_path = os.path.join(
            output_path, self.base_model.name, f'output-{input_video_fn.replace(".mp4","")}.mp4'
        )
        input_frames_path = os.path.join(
            input_path, self.base_model.name, "frames", f"{input_video_fn}_frames.npy.gz"
        )
        with open(labels_path, "r") as f:
            labels_json = json.load(f)
        video_stream = ffmpeg.probe(input_video_path)["streams"][0]
        ns = {"__builtins__": None}
        # frame_height = int(video_stream["height"])
        # frame_width = int(video_stream["width"])
        fps = math.ceil(float(eval(video_stream["avg_frame_rate"], ns)))
        ffmpeg_input_args = {
            "hide_banner": None,
            "loglevel": "error",
            "stats": None,
            "v": "error",
        }
        generated_frames = os.path.exists(input_frames_path)
        generated_video = os.path.exists(output_video_path)
        if generated_frames:
            if generated_video:
                print(f"{RED}already generated frames and video for video {input_video_fn}{RESET}")
            else:
                print(f"{RED}reading generated frames for video {input_video_fn}{RESET}")
                with gzip.GzipFile(input_frames_path, "r") as f:
                    frames = np.load(f)
                print(f"{RED}writing output video {output_video_path}{RESET}")
                vidwrite(
                    output_video_path,
                    frames,
                    fps=fps // 4,
                    in_pix_fmt="rgb24",
                    input_args=ffmpeg_input_args,
                )
                return
        else:
            print(f"{RED}generating frames for video {input_video_fn}{RESET}")
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
                    label = labels_json[
                        str(np.argmax(self.model.predict(expanded_frame, verbose=1)[1]))
                    ]
                    # print(label)
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
                    # showing the output image
                    # plt.imshow(np.array(image))
                    # plt.show()
                    frames.append(np.array(image, dtype=np.uint8))
                    count += fps // 2  # i.e. at 30 fps, this advances one second
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
                else:
                    vidcap.release()
                    break
            frames = np.array(frames)
            print(f"{RED}saving frames in {input_frames_path}{RESET}")
            with gzip.GzipFile(input_frames_path, "w") as f:
                np.save(f, frames)
            print(f"{RED}writing output video {output_video_path}{RESET}")
            vidwrite(
                output_video_path,
                frames,
                fps=fps // 4,
                in_pix_fmt="rgb24",
                input_args=ffmpeg_input_args,
            )
            return
