import argparse
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

from modules.config import logger

logger.info("$BOLD$GREENStart: $BLUE%s", datetime.now().strftime("%d/%m/%Y, %T"))

import pandas as pd

start = timer()
from keras.applications import (
    VGG16,
    VGG19,
    ResNet50,
    ResNet50V2,
    ResNet152,
    ResNet152V2,
)

stop = timer()
logger.info(
    f"$BOLD$BLUEImporting `keras.applications` took $RED{stop-start:.2f}$BLUE seconds"
)
from matplotlib import pyplot as plt

from modules.config import input_videos_filenames, output_path
from modules.custom_model import CustomModel


class HelpAction(argparse.Action):
    def __call__(self, parser, *args, **kwargs):
        parser.print_help()
        stop = timer()
        logger.info(
            f"$BOLD$BLUEProgram execution took $RED{stop-start:.2f}$BLUE seconds"
        )
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Program that trains and tests a model for road sign detection",
        add_help=False,
    )
    parser.add_argument("-h", "--help", nargs=0, action=HelpAction)
    parser.add_argument(
        "--benchmark", action="store_true", help="benchmark base models"
    )
    parser.add_argument(
        "--random-images",
        action="store_true",
        help="select random images from dataset for testing",
    )
    parser.add_argument(
        "--test", action="store_true", help="test model on both images and videos"
    )
    parser.add_argument(
        "--test-images", action="store_true", help="test model on images"
    )
    parser.add_argument(
        "--test-videos", action="store_true", help="test model on videos"
    )
    parser.add_argument(
        "--lite", action="store_true", help="convert HDF5 model to `tf.lite` model"
    )
    args = parser.parse_args()
    if args.test:
        args.test_images = True
        args.test_videos = True
    if args.benchmark:
        models = {
            "resnet152": ResNet152,
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
        }
    else:
        models = {
            "vgg16": VGG16,
        }
    for name, model in models.items():
        logger.info(f"$BLUE$BOLDBase model: $RED{name}$RESET")

        saved_model_path: str = os.path.join(output_path, name, "model.h5")
        trained: bool = os.path.exists(saved_model_path)

        custom_model_instance = CustomModel(
            base_model_function=model, trained=trained, lite_model_required=args.lite
        )

        if args.test_images:
            custom_model_instance.test_model_images(only_random=args.random_images)
        if args.test_videos:
            for input_filename in input_videos_filenames:
                custom_model_instance.test_model_videos(input_video_fn=input_filename)
        if args.benchmark:
            custom_model = custom_model_instance.model
            history = custom_model_instance.history

            model_benchmarks["model_name"].append(name)
            model_benchmarks["num_model_params"].append(custom_model.count_params())
            model_benchmarks["label_validation_accuracy"].append(
                float(history["val_class_label_accuracy"][-1]) * 100
            )
    if args.benchmark:
        benchmark_df = pd.DataFrame(model_benchmarks)
        benchmark_df.sort_values(
            "label_validation_accuracy", inplace=True, ascending=False
        )
        benchmark_df["label_validation_accuracy"] = benchmark_df[
            "label_validation_accuracy"
        ].transform(lambda x: f"{x:.2f}%")
        benchmark_df.to_csv(os.path.join(output_path, "benchmark_df.csv"), index=False)
        logger.info(f"$BOLD$BLUE{benchmark_df.to_string()}$RESET")

        # save plot to file
        markers = [".", ",", "o", "v", "^", "<", ">", "*", "+", "|", "_"]
        plt.figure(figsize=(10, 8))
        for row in benchmark_df.itertuples():
            plt.scatter(
                x=row.num_model_params,
                y=row.label_validation_accuracy,
                # y=row.random_accuracy,
                label=row.model_name,
                marker=markers[row.Index],
                s=150,
                linewidths=2,
            )
        plt.xscale("log")
        plt.xlabel("Number of Parameters in Model")
        plt.ylabel("Validation Accuracy after 10 Epochs")
        plt.title("Accuracy vs Model Size")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path + "/plot.png")


if __name__ == "__main__":
    main()
    stop = timer()
    logger.info(f"$BOLD$BLUEProgram execution took $RED{stop-start:.2f}$BLUE seconds")
