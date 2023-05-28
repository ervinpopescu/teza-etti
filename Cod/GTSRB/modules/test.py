import gzip
import json
import math
import os
import pickle
import random
import subprocess

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from keras.models import Model
from keras.utils import img_to_array, load_img
from matplotlib import pyplot as plt
from modules import (
    BATCH_SIZE,
    IMG_SIZE,
    input_path,
    input_videos_dir,
    labels_path,
    output_path,
    predicted_labels_path,
    scores_path,
    test_data_dir,
)
from modules.load_data import load_test_data
from PIL import Image, ImageDraw, ImageFont


class VideoWriter:
    def __init__(
        self,
        fn,
        vcodec="libx264",
        fps=60,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.fn = fn
        self.process: subprocess.Popen = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = vcodec

    def add(self, frame: np.ndarray):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(w, h),
                    **self.input_args,
                )
                .output(self.fn, **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            # print(self.process.args)
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def vidwrite(fn, images, **kwargs):
    writer = VideoWriter(fn, **kwargs)
    for image in images:
        writer.add(image)
    writer.close()


def test_model(model: Model):
    images, bboxes, image_paths = load_test_data(test_data_dir)
    if os.path.exists(predicted_labels_path):
        with open(predicted_labels_path, "rb") as f:
            predicted_labels = pickle.load(f)
    else:
        predicted_labels = model.predict(images, batch_size=BATCH_SIZE, verbose=1)[1]
        with open(predicted_labels_path, "wb") as f:
            pickle.dump(predicted_labels, f)
    predicted_labels = np.array(predicted_labels)
    with open(os.path.join(test_data_dir, "Test.csv")) as f:
        correct_labels = pd.read_csv(f, sep=",")["ClassId"].to_numpy(dtype="uint32")
    with open(labels_path, "r") as f:
        labels_json = json.load(f)

    testTargets = {"class_label": predicted_labels, "bounding_box": bboxes}
    metrics_names = model.metrics_names

    if not os.path.exists(scores_path):
        # Evaluate the model
        scores = model.evaluate(
            images,
            testTargets,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
        with open(scores_path, "w") as f:
            for name, score in zip(metrics_names, scores):
                if "loss" in name:
                    line = "{}: {:.2f}\n".format(name, score)
                else:
                    line = "{}: {:.2f}%\n".format(name, score * 100)
                print(line.strip("\n"))
                f.write(line)
    else:
        with open(scores_path, "r") as f:
            scores = f.read().splitlines()
            for score in scores:
                print(score.strip("\n"))

    correct = 0

    if not os.path.exists(scores_path):
        for image_path in image_paths:
            index = np.where(image_paths == image_path)[0][0]
            image = load_img(image_path, target_size=IMG_SIZE)
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            # # finding class label with highest pred. probability
            i = np.argmax(predicted_labels[index], axis=0)
            predicted_label = labels_json[str(i)]
            correct_label = labels_json[str(correct_labels[index])]
            # print("\n")
            # print(f"{index}: {image_path}")
            # print(f"Predicted label: {predicted_label}")
            # print(f"Correct label: {correct_label}")

            if predicted_label == correct_label:
                correct += 1

            with open(scores_path, "w") as f:
                f.write(f"Test accuracy: {correct/len(images)*100:.2f}%")
            print(f"Test accuracy: {correct/len(images)*100:.2f}%")

    random_choices = random.choices(image_paths, k=10)
    for image_path in random_choices:
        index = np.where(image_paths == image_path)[0][0]
        i = np.argmax(predicted_labels[index], axis=0)
        predicted_label = labels_json[str(i)]
        correct_label = labels_json[str(correct_labels[index])]
        if predicted_label == correct_label:
            correct += 1

        image = Image.open(image_path)
        image = Image.Image.resize(image, size=(256, 256))

        # scaling pred. bbox coords according to image dims
        (xmin, ymin, xmax, ymax) = bboxes[index]
        (h, w) = (image.height, image.width)
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)

        # drawing bbox and label on image
        draw = ImageDraw.ImageDraw(image, "RGBA")
        draw.font = ImageFont.truetype(
            "/usr/share/fonts/OTF/intelone-mono-font-family-regular.otf", size=13
        )
        draw.fontmode = "L"
        draw.text((xmin, (ymax - 10) / 2), predicted_label, (0, 255, 0))

        draw.rectangle(
            xy=(
                (xmin, ymax),
                (xmax, ymin),
            ),
            fill=(0, 0, 0, 0),
            outline=(0, 255, 0),
        )

        # showing the output image
        plt.imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
        plt.show()

    # print(f"correct labels for random images: {correct}/10")

    return


def test_model_video(model: Model, input_video_fn: str):
    print(f"processing input video {input_video_fn}")
    input_video_path = os.path.join(input_videos_dir, input_video_fn)
    output_video_path = os.path.join(
        output_path, f'output-{input_video_fn.replace(".mp4","")}.mp4'
    )
    input_frames_path = os.path.join(
        input_path, "frames", f"{input_video_fn}_frames.npy.gz"
    )
    with open(labels_path, "r") as f:
        labels_json = json.load(f)
    video_stream = ffmpeg.probe(input_video_path)["streams"][0]
    ns = {"__builtins__": None}
    frame_height = int(video_stream["height"])
    frame_width = int(video_stream["width"])
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
            print(f"already generated frames and video for video {input_video_fn}")
        else:
            print(f"reading generated frames for video {input_video_fn}")
            with gzip.GzipFile(input_frames_path, "r") as f:
                frames = np.load(f)
            print(f"writing output video {output_video_path}")
            vidwrite(
                output_video_path,
                frames,
                fps=fps // 4,
                in_pix_fmt="rgb24",
                input_args=ffmpeg_input_args,
            )
            return
    else:
        print(f"generating frames for video {input_video_fn}")
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
                    str(np.argmax(model.predict(expanded_frame, verbose=0)[1]))
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
        print(f"saving frames in {input_frames_path}")
        with gzip.GzipFile(input_frames_path, "w") as f:
            np.save(f, frames)
        print(f"writing output video {output_video_path}")
        vidwrite(
            output_video_path,
            frames,
            fps=fps // 4,
            in_pix_fmt="rgb24",
            input_args=ffmpeg_input_args,
        )
        return
