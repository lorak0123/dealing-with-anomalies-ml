import os
from pathlib import Path


def prepare_directory(path: Path) -> Path:
    """
    Creates the directory if it does not exist.

    Args:
        path: The path to the directory.

    Returns:
        The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
