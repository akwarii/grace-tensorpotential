"""Plain-text formatting helpers that need neither TensorFlow nor any other heavy import."""

from __future__ import annotations

from pathlib import Path

_BINARY_PREFIXES = ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi")


def sizeof_fmt(file_name_or_size: str | float, suffix: str = "B") -> str:
    """Format a size in bytes with a binary prefix and one decimal.

    Parameters
    ----------
    file_name_or_size : str or float
        A size in bytes, or the path of a file whose size is read from disk.
    suffix : str, default "B"
        Text appended after the prefix.

    Returns
    -------
    str
        The size divided by 1024 until it is below 1024, for example ``"1.5KiB"``;
        sizes of 1024**8 or more stay in ``Yi``.
    """
    if isinstance(file_name_or_size, str):
        file_name_or_size = Path(file_name_or_size).stat().st_size
    for unit in _BINARY_PREFIXES:
        if abs(file_name_or_size) < 1024.0:
            return f"{file_name_or_size:3.1f}{unit}{suffix}"
        file_name_or_size /= 1024.0
    return f"{file_name_or_size:.1f}Yi{suffix}"
