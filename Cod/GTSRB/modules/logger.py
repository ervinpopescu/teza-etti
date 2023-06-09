import warnings
from logging import (
    Formatter,
    LogRecord,
    Logger,
    FileHandler,
    captureWarnings,
)


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
    format_str: str = "$BOLD$COLOR==> $BOLD$BLUE%(message)s",
    log_level="INFO",
) -> None:
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = FileHandler(filename=log_path, mode="w")
    formatter: Formatter = ColorFormatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    # Capture everything from the warnings module.
    captureWarnings(True)
    warnings.simplefilter("always")