from timeit import default_timer as timer

start = timer()

from logging import Logger, getLogger
import os
import pathlib
from modules.custom_model import CustomModel
from modules.config import RED, BLUE
from modules.logger import init_log

logger: Logger = getLogger("poc")
init_log(
    logger,
    log_path=os.path.join(pathlib.Path(__file__).parent.resolve(), "poc.log"),
    format_str="$BOLD$COLOR==> $BOLD$BLUE%(message)s",
    log_level="INFO",
)


def main():
    logger.info("hello world")


if __name__ == "__main__":
    main()
    stop = timer()
    logger.info(f"Program execution took {RED}{stop-start:.6f}{BLUE} seconds")
