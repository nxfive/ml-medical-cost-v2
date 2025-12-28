import logging
from pathlib import Path

from src.io.file_ops import PathManager
from src.io.readers import YamlReader
from src.settings import Settings


def setup_logging() -> logging.Logger:
    """
    Configure logging from a YAML config file, using logging level from Settings.
    Wraps logger with RequestLoggerAdapter in dev environment.
    """
    log_config_file = Path(__file__).parent / "config.yaml"
    log_level = Settings.logging_level()

    if PathManager.exists(log_config_file):
        config = YamlReader().read(log_config_file)

        if "loggers" in config and "app" in config["loggers"]:
            config["loggers"]["app"]["level"] = log_level
        if "handlers" in config and "console" in config["handlers"]:
            config["handlers"]["console"]["level"] = log_level

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    return logging.getLogger("app")


logger = setup_logging()
