import os
import pathlib

RED = "\033[1;31m"
GREEN = "\033[1;32m"
BLUE = "\033[1;34m"
RESET = "\033[0m"

main_file_path = pathlib.Path(__file__).parent.parent
input_path = os.path.join(main_file_path, "input")
output_path = os.path.join(main_file_path, "output")


# Define the location of the dataset
training_data_dir = os.path.join(input_path, "images", "Training")
test_data_dir = os.path.join(input_path, "images", "Test")
input_videos_dir = os.path.join(input_path, "videos")

# Define the image size and number of classes
IMG_SIZE = (64, 64)
VIDEO_SIZE = (1024, 1024)
NUM_CLASSES = 43
INIT_LR = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 64

labels_path = os.path.join(input_path, "labels.json")
input_videos_filenames = os.listdir(input_videos_dir)
