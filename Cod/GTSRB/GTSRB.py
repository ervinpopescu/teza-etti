import os

from keras.models import Model, load_model
from modules.config import define_model, input_videos_filenames, saved_model_path
from modules.test import test_model, test_model_video
from modules.train import train_model


def main():
    not_trained: bool = not os.path.exists(saved_model_path)
    if not_trained:
        model: Model = define_model()
        train_model(model)
        test_model(model)
        for input_filename in input_videos_filenames:
            test_model_video(model, input_video_fn=input_filename)
    else:
        model = load_model(saved_model_path)
        # test_model(model)
        for input_filename in input_videos_filenames:
            test_model_video(model, input_video_fn=input_filename)


if __name__ == "__main__":
    main()
