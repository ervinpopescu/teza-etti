import argparse
import os

import jsonpickle
import tqdm
from keras.applications import (
    VGG16,
    VGG19,
    InceptionResNetV2,
    InceptionV3,
    MobileNetV3Large,
    MobileNetV3Small,
    ResNet50,
    ResNet50V2,
    ResNet152V2,
    Xception,
)
from keras.models import Model, load_model
from modules.config import output_path  # input_videos_filenames,
from modules.custom_model import CustomModel  # define_model,; saved_model_path,

# from modules.test import test_model_images, test_model_videos
# from modules.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Program that trains and tests a model for road sign detection"
    )
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument(
        "--test", action="store_true", help="test model on images and videos"
    )
    parser.add_argument(
        "--test-images", action="store_true", help="test model on images"
    )
    parser.add_argument(
        "--test-videos", action="store_true", help="test model on videos"
    )
    args = parser.parse_args()
    if args.test:
        args.test_images = True
        args.test_videos = True
    models = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        # "inception_resnet_v2": InceptionResNetV2,
        # "inception_v3": InceptionV3,
        "MobilenetV3large": MobileNetV3Large,
        "MobilenetV3small": MobileNetV3Small,
        "resnet152v2": ResNet152V2,
        "resnet50": ResNet50,
        "resnet50v2": ResNet50V2,
        "xception": Xception,
    }
    model_benchmarks = {
        "model_name": [],
        "num_model_params": [],
        "label_validation_accuracy": [],
    }
    GREEN = "\033[1;32m"
    RESET = "\033[0m"
    for name, model in models.items():
        print(f"{GREEN}Model: {name}{RESET}")
        model: Model
        saved_model_path = os.path.join(output_path, name, "model.h5")
        history_path = os.path.join(output_path, name, "training_history.jsonpickle")
        trained: bool = os.path.exists(saved_model_path)
        if not trained:
            if args.train:
                custom_model_instance = CustomModel(base_model_function=model)
                custom_model = custom_model_instance.model
                pretrained_model = custom_model_instance.base_model
                history = custom_model_instance.train()
            # if args.test_images:
            #     test_model_images(custom_model)
            # if args.test_videos:
            #     for input_filename in input_videos_filenames:
            #         test_model_videos(custom_model, input_video_fn=input_filename)
        else:
            custom_model = load_model(saved_model_path)
            with open(history_path, "r") as f:
                history = jsonpickle.decode(f.read())
            # if args.test_images:
            #     test_model_images(model)
            # if args.test_videos:
            #     for input_filename in input_videos_filenames:
            #         test_model_videos(model, input_video_fn=input_filename)
        print(history.history.keys())
        # Calculate all relevant metrics
        model_benchmarks["model_name"].append(name)
        model_benchmarks["num_model_params"].append(custom_model.count_params())
        model_benchmarks["label_validation_accuracy"].append(
            history.history["val_class_label_accuracy"][-1]
        )


if __name__ == "__main__":
    main()
