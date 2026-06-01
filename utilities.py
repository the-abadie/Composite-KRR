import logging


def configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(level=level, format="[%(name)s]: %(message)s")


def time_dif(start:float, end:float) -> str:
    dt:float = end - start
    if dt < 1.0: # seconds
        return(f"{int(1000*dt)} ms")
    return(f"{dt:.3f} sec")
