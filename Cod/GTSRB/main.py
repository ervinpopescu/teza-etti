import argparse
import os

import jsonpickle
import pandas as pd
from keras.applications import (
    VGG16,
    VGG19,
    MobileNetV3Large,
    MobileNetV3Small,
    ResNet50,
    ResNet50V2,
    ResNet152V2,
)
from matplotlib import pyplot as plt
from modules.config import input_videos_filenames, output_path
from modules.custom_model import CustomModel


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
        "MobilenetV3large": MobileNetV3Large,
        "MobilenetV3small": MobileNetV3Small,
        "resnet152v2": ResNet152V2,
        "resnet50": ResNet50,
        "resnet50v2": ResNet50V2,
        "vgg16": VGG16,
        "vgg19": VGG19,
    }
    model_benchmarks = {
        "model_name": [],
        "num_model_params": [],
        "label_validation_accuracy": [],
        # "random_accuracy": [],
    }
    GREEN = "\033[1;32m"
    RESET = "\033[0m"
    for name, model in models.items():
        # print(f"{GREEN}Model: {name}{RESET}")
        saved_model_path = os.path.join(output_path, name, "model.h5")
        history_path = os.path.join(output_path, name, "training_history.json")
        trained: bool = os.path.exists(saved_model_path)
        if not trained:
            if args.train:
                custom_model_instance = CustomModel(
                    base_model_function=model, trained=trained
                )
                custom_model = custom_model_instance.model
                history = custom_model_instance.train()
                with open(custom_model_instance.history_path, "w") as f:
                    f.write(jsonpickle.encode(history.history))
            if args.test_images:
                custom_model_instance.test_model_images()
                # random_acc = custom_model_instance.test_model_images()
            if args.test_videos:
                for input_filename in input_videos_filenames:
                    custom_model_instance.test_model_videos(
                        input_video_fn=input_filename
                    )
        else:
            custom_model_instance = CustomModel(
                base_model_function=model, trained=trained
            )
            custom_model = custom_model_instance.model
            with open(history_path, "r") as f:
                history = jsonpickle.decode(f.read())
            if args.test_images:
                custom_model_instance.test_model_images()
                # random_acc = custom_model_instance.test_model_images()
            if args.test_videos:
                for input_filename in input_videos_filenames:
                    custom_model_instance.test_model_videos(
                        input_video_fn=input_filename
                    )
        # Calculate all relevant metrics
        model_benchmarks["model_name"].append(name)
        model_benchmarks["num_model_params"].append(custom_model.count_params())
        model_benchmarks["label_validation_accuracy"].append(
            history["val_class_label_accuracy"][-1]
        )
        # model_benchmarks["random_accuracy"].append(random_acc)
        # Convert Results to DataFrame for easy viewing
    benchmark_df = pd.DataFrame(model_benchmarks)

    # sort in ascending order of num_model_params column
    benchmark_df.sort_values("label_validation_accuracy", inplace=True)
    # benchmark_df["label_validation_accuracy"].apply(lambda x: "{0:.2f}".format(x * 100))

    # write results to csv file
    benchmark_df.to_csv(output_path + "/benchmark_df.csv", index=False)
    print(benchmark_df)
    # markers = [".", ",", "o", "v", "^", "<", ">", "*", "+", "|", "_"]
    # plt.figure(figsize=(7, 5))

    # for row in benchmark_df.itertuples():
    #     plt.scatter(
    #         x=row.num_model_params,
    #         y=row.label_validation_accuracy,
    #         # y=row.random_accuracy,
    #         label=row.model_name,
    #         marker=markers[row.Index],
    #         s=150,
    #         linewidths=2,
    #     )

    # plt.xscale("log")
    # plt.xlabel("Number of Parameters in Model")
    # plt.ylabel("Validation Accuracy after 10 Epochs")
    # plt.title("Accuracy vs Model Size")

    # # Move legend out of the plot
    # plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # plt.show()


if __name__ == "__main__":
    main()
