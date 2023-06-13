import argparse
import json
import multiprocessing
import os
import pathlib
from datetime import datetime
from logging import Logger, getLogger
import random
from time import sleep
from timeit import default_timer as timer

from modules.logger import init_log

logger: Logger = getLogger("poc")
current_file_path = pathlib.Path(__file__).parent.resolve()
init_log(
    logger,
    maxBytes=100000,
    mode="a",
    log_path=os.path.join(current_file_path, "poc.log"),
    format_str="%(message)s",
    log_level="INFO",
)
logger.info("$BOLD$GREENStart: $BLUE%s", datetime.now().strftime("%d/%m/%Y, %T"))


import numpy as np
import pandas as pd

start = timer()
from keras.models import load_model

from modules.config import BATCH_SIZE, labels_path, output_path, test_data_dir
from modules.load_data import load_test_data_poc

# from keras.models import Model


logger.info(f"$BLUEImporting took $RED{timer()-start:.2f}$BLUE seconds")


def main():
    parser = argparse.ArgumentParser(prog="POC", description="Proof of concept")
    parser.add_argument("--iterations", type=int, nargs=1, default=10)
    parser.add_argument("--images", type=int, nargs=1, default=50)
    args = parser.parse_args()
    n_img = args.images[0]
    n_iter = args.iterations[0]
    with open(os.path.join(test_data_dir, "Test.csv")) as f:
        correct_labels = pd.read_csv(f, sep=",")["ClassId"].to_numpy(dtype="uint32")
    with open(labels_path, "r") as f:
        labels_json = json.load(f)
    start = timer()
    indexes, images = load_test_data_poc(
        data_dir=test_data_dir, size=n_img, seed="Laura"
    )
    logger.info(f"$BLUELoading images took $RED{timer()-start:.2f}$BLUE seconds")
    # model_names = sorted(
    #     [
    #         i
    #         for i in os.listdir(output_path)
    #         if os.path.isdir(os.path.join(output_path, i))
    #     ]
    # )
    model_names = ["vgg16"]
    for name in model_names:
        logger.info(f"$BLUEModel: $RED{name}")
        start = timer()
        model = load_model(os.path.join(output_path, name, "model.h5"))
        logger.info(f"\t$BLUELoading model took $RED{timer()-start:.2f}$BLUE seconds\n")
        logger.info(f"\t$BLUE# images: $RED{n_img}")
        logger.info(f"\t$BLUE# iterations: $RED{n_iter}")
        model_accuracies = []
        predicted_labels = model.predict(
            images,
            batch_size=BATCH_SIZE,
            verbose=0,
            # workers=12,
            # use_multiprocessing=True,
        )[1]
        for i in range(n_iter):
            correct = 0
            for count, index in zip(range(n_img), indexes):
                i = np.argmax(predicted_labels[count], axis=0)
                predicted_label = labels_json[str(i)]
                correct_label = labels_json[str(correct_labels[index])]
                if predicted_label == correct_label:
                    correct += 1
            random_acc = float(f"{correct/n_img*100:.2f}")
            model_accuracies.append(random_acc)
        logger.info(f"\t$BLUETest accuracy: $RED{np.average(model_accuracies):.2f}%")
        logger.info(f"\t$BLUETesting took $RED{timer()-start:.2f}$BLUE seconds\n")
    for i in range(50):
        # n_img = int(input("gib # imagez plz: "))
        # n_iter = int(input("gib # iterz plz: "))
        random.seed(random.random())
        start=timer()
        n_img = random.randint(50,1000)
        n_iter = random.randint(10,50)
        indexes, images = load_test_data_poc(
            data_dir=test_data_dir, size=n_img, seed="Laura"
        )
        logger.info(f"\t$BLUE# images: $RED{n_img}")
        logger.info(f"\t$BLUE# iterations: $RED{n_iter}")
        model_accuracies = []
        predicted_labels = model.predict(
            images,
            batch_size=BATCH_SIZE,
            verbose=0,
            workers=12,
            use_multiprocessing=True,
        )[1]
        for i in range(n_iter):
            correct = 0
            for count, index in zip(range(n_img), indexes):
                i = np.argmax(predicted_labels[count], axis=0)
                predicted_label = labels_json[str(i)]
                correct_label = labels_json[str(correct_labels[index])]
                if predicted_label == correct_label:
                    correct += 1
            random_acc = float(f"{correct/n_img*100:.2f}")
            model_accuracies.append(random_acc)
        logger.info(
            f"\t$BLUETest accuracy: $RED{np.average(model_accuracies):.2f}%"
        )
        logger.info(f"\t$BLUETesting took $RED{timer()-start:.2f}$BLUE seconds\n")
        sleep(2)


if __name__ == "__main__":
    main()
    stop = timer()
    # logger.info(f"$BLUEProgram execution took $RED{stop-start:.2f}$BLUE seconds")
    # logger.info("$GREENEnd: $BLUE%s", datetime.now().strftime("%d/%m/%Y, %T"))
