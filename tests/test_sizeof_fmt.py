"""Characterization of ``sizeof_fmt``, the byte-size formatter copied into four modules (QUAL2).

The expected strings are worked out by hand from the rule "divide by 1024 until the value is below
1024, print one decimal and the binary prefix". There is no physics in this function.
"""

from __future__ import annotations

import importlib

import pytest

# Modules that define or import the formatter; all of them must give the same strings.
MODULES = (
    "tensorpotential.cli.data",
    "tensorpotential.scripts.df2extxyz",
    "tensorpotential.scripts.extxyz2df",
    "tensorpotential.scripts.grace_preprocess",
)


@pytest.fixture(params=MODULES)
def sizeof_fmt(request):
    return importlib.import_module(request.param).sizeof_fmt


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0.0B"),
        (1, "1.0B"),
        (1023, "1023.0B"),
        (1024, "1.0KiB"),
        (1536, "1.5KiB"),
        (1024**2, "1.0MiB"),
        (5 * 1024**3, "5.0GiB"),
        (1024**4, "1.0TiB"),
        (1024**5, "1.0PiB"),
        (1024**6, "1.0EiB"),
        (1024**7, "1.0ZiB"),
        (1024**8, "1.0YiB"),
        (1024**9, "1024.0YiB"),
        (-2048, "-2.0KiB"),
        (-1023, "-1023.0B"),
        (2.5, "2.5B"),
    ],
)
def test_size_in_bytes(sizeof_fmt, size, expected):
    assert sizeof_fmt(size) == expected


def test_suffix_replaces_the_trailing_b(sizeof_fmt):
    assert sizeof_fmt(1024, suffix="bit") == "1.0Kibit"
    assert sizeof_fmt(7, suffix="") == "7.0"
    assert sizeof_fmt(1024**9, suffix="b") == "1024.0Yib"


def test_the_width_of_the_first_field_does_not_pad_the_result(sizeof_fmt):
    assert sizeof_fmt(5) == "5.0B"


@pytest.mark.parametrize(
    ("nbytes", "expected"), [(0, "0.0B"), (2048, "2.0KiB"), (1500, "1.5KiB")]
)
def test_a_path_gives_the_size_of_the_file(sizeof_fmt, tmp_path, nbytes, expected):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * nbytes)
    assert sizeof_fmt(str(path)) == expected


def test_a_missing_file_raises(sizeof_fmt, tmp_path):
    with pytest.raises(FileNotFoundError):
        sizeof_fmt(str(tmp_path / "absent.bin"))
