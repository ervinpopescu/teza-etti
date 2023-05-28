import argparse
import os

from keras.models import Model, load_model
from modules.config import define_model, input_videos_filenames, saved_model_path
from modules.test import test_model_images, test_model_videos
from modules.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Program that trains and tests a model for road sign detection"
    )
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="test model on images and videos")
    parser.add_argument("--test-images", action="store_true", help="test model on images")
    parser.add_argument("--test-videos", action="store_true", help="test model on videos")
    args = parser.parse_args()
    if args.test:
        args.test_images = True
        args.test_videos = True
    trained: bool = os.path.exists(saved_model_path)
    if not trained:
        if args.train:
            model: Model = define_model()
            train_model(model)
        if args.test_images:
            test_model_images(model)
        if args.test_videos:
            for input_filename in input_videos_filenames:
                test_model_videos(model, input_video_fn=input_filename)
    else:
        model = load_model(saved_model_path)
        if args.test_images:
            test_model_images(model)
        if args.test_videos:
            for input_filename in input_videos_filenames:
                test_model_videos(model, input_video_fn=input_filename)


if __name__ == "__main__":
    main()
