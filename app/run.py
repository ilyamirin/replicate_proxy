from copy import deepcopy

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from app.config import load_settings


def _build_log_config(level: str) -> dict:
    config = deepcopy(LOGGING_CONFIG)
    config["root"] = {"handlers": ["default"], "level": level}
    config["loggers"]["app"] = {
        "handlers": ["default"],
        "level": level,
        "propagate": False,
    }
    return config


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level.lower(),
        log_config=_build_log_config(settings.app_log_level),
    )


if __name__ == "__main__":
    main()
