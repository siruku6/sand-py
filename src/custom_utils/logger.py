# ---------------------------------------------------
# Logger の設定を行うためのモジュール
# ---------------------------------------------------
import os
from logging import Logger, config, getLogger

from custom_utils.paths import APP_ROOT_DIR

LOG_PATH: str = os.path.join(APP_ROOT_DIR, "logs", "logger.log")


log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": (
                "%(asctime)s [%(levelname)s]: %(message)s :"
                "%(name)s %(pathname)s.%(funcName)s:%(lineno)s"
            )
        }
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "fileHandler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": LOG_PATH,
            "maxBytes": 1024 * 1024 * 32,
            "backupCount": 3,
        },
    },
    "loggers": {
        "__main__": {
            "level": "DEBUG",
            # "handlers": ["consoleHandler", "fileHandler"],
            "handlers": ["fileHandler"],
            "propagate": False,
        }
    },
}


def init_logger() -> Logger:
    """
    Usage
    ------
    >>> from custom_utils.logger import init_logger
    >>> logger = init_logger()
    >>> logger.debug("This is debug message.")
    """
    # NOTE: logger 設定をファイルで書きたい場合は以下のようにしてもいいかも
    # with open(os.path.join(APP_ROOT_DIR, log_config.json), "r") as f:
    #     log_config = json.load(f)

    os.makedirs(os.path.join(APP_ROOT_DIR, "logs"), exist_ok=True)
    config.dictConfig(log_config)

    # "loggers" から logger 名を 1 つ選ぶ
    logger: Logger = getLogger("__main__")
    return logger
