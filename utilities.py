import logging
from config import VERBOSITY
from pathlib import Path

def configure_logging(verbosity: int, log_path=None) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter("[%(name)s]: %(message)s")
    _quiet_third_party_loggers()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in root.handlers):
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)


def _quiet_third_party_loggers() -> None:
    for logger_name in (
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def time_dif(start:float, end:float) -> str:
    dt:float = end - start
    if dt < 1.0: # seconds
        return(f"{int(1000*dt)} ms")
    return(f"{dt:.3f} sec")
