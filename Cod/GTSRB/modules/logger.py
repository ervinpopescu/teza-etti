import os
import warnings
from logging import Formatter, Logger, LogRecord, captureWarnings, getLogger
from logging.handlers import RotatingFileHandler


class ColorFormatter(Formatter):
    """Logging formatter adding console colors to the output."""

    black, red, green, yellow, blue, magenta, cyan, white = range(8)
    colors = {
        "WARNING": yellow,
        "INFO": green,
        "DEBUG": blue,
        "CRITICAL": yellow,
        "ERROR": red,
        "RED": red,
        "GREEN": green,
        "YELLOW": yellow,
        "BLUE": blue,
        "MAGENTA": magenta,
        "CYAN": cyan,
        "WHITE": white,
    }
    reset_seq = "\033[0m"
    color_seq = "\033[%dm"
    bold_seq = "\033[1m"

    def format(self, record: LogRecord) -> str:
        """Format the record with colors."""
        color = self.color_seq % (30 + self.colors[record.levelname])
        message = Formatter.format(self, record)
        message = (
            message.replace("$RESET", self.reset_seq)
            .replace("$BOLD", self.bold_seq)
            .replace("$COLOR", color)
        )
        for color, value in self.colors.items():
            message = (
                message.replace("$" + color, self.color_seq % (value + 30))
                .replace("$BG" + color, self.color_seq % (value + 40))
                .replace("$BG-" + color, self.color_seq % (value + 40))
            )
        return message + self.reset_seq


def init_log(
    logger: Logger,
    log_path: str,
    maxBytes: int = 10000,
    backupCount: int = 0,
    mode: str = "a",
    format_str: str = "%(message)s",
    log_level="INFO",
) -> None:
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # should_roll_over = os.path.exists(log_path)
    handler = RotatingFileHandler(
        filename=log_path,
        mode=mode,
        backupCount=backupCount,
        maxBytes=maxBytes,
    )
    # if should_roll_over:
    #     handler.doRollover()
    formatter: Formatter = ColorFormatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    warnings_logger = getLogger("py.warnings")
    warnings_logger.addHandler(handler)
    # warnings_logger.setLevel("CRITICAL")
    # Capture everything from the warnings module.
    captureWarnings(True)
    warnings.simplefilter("ignore")
