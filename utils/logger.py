import logging
from logging import handlers

import sys
from rich.console import Console
from rich.logging import RichHandler

level_relations = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "crit": logging.CRITICAL,
}


def create_logger(
    filename, level="info", fmt="{} %(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(level_relations.get(level))

    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)

    th = handlers.RotatingFileHandler(filename=filename, encoding="utf-8")
    th.setFormatter(format_str)
    logger.addHandler(th)

    return logger


class Logger(logging.Logger):
    def __init__(
        self,
        log_path,
        name="root",
        level="info",
        rich_fmt="%(asctime)s - %(message)s",
        file_fmt="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    ):
        logging.Logger.__init__(self, name)

        self.setLevel(level_relations.get(level))

        # Rich handler for stdout
        console = Console(file=sys.stdout)
        rich_handler = RichHandler(console=console, rich_tracebacks=True)
        rich_handler.setFormatter(logging.Formatter(fmt=rich_fmt, datefmt="[%X]"))
        self.addHandler(rich_handler)

        # File handler
        filename = f"{log_path}/log.txt"
        file_handler = handlers.RotatingFileHandler(filename=filename, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=file_fmt))
        self.addHandler(file_handler)


if __name__ == "__main__":
    log = create_logger("all.log", level="debug")
    log.debug("debug")
    log.info("info")
