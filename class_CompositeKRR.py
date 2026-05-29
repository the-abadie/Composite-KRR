import logging

from config import VERBOSITY
from utilities import configure_logging

configure_logging(VERBOSITY)
logger = logging.getLogger("class_CompositeKRR")
