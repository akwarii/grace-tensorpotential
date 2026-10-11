"""Tests of ``calculator/foundation_models.py``: the extraction of a downloaded model archive.

A foundation-model archive comes from the network, so ``safe_extract`` is a trust boundary. Logic layer only (no
physics here): what it accepts, what it refuses, and where the files land. Archives are built in ``tmp_path``.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tensorpotential.calculator.foundation_models import safe_extract


def _add_file(tar: tarfile.TarFile, name: str, payload: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _add_dir(tar: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    tar.addfile(info)


def _archive(tmp_path: Path, build) -> Path:
    """Write ``model.tar.gz`` whose members are added by ``build(tar)``."""
    path = tmp_path / "model.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return path


def test_extracts_one_top_level_folder_and_returns_its_name(tmp_path):
    def build(tar):
        _add_dir(tar, "GRACE-model")
        _add_file(tar, "GRACE-model/model.yaml", b"a: 1\n")
        _add_dir(tar, "GRACE-model/sub")
        _add_file(tar, "GRACE-model/sub/ckpt.index", b"abc")

    archive = _archive(tmp_path, build)
    target = tmp_path / "out"
    target.mkdir()

    assert safe_extract(archive, path=target) == "GRACE-model"
    assert (target / "GRACE-model" / "model.yaml").read_bytes() == b"a: 1\n"
    assert (target / "GRACE-model" / "sub" / "ckpt.index").read_bytes() == b"abc"


def test_extracts_only_the_requested_members(tmp_path):
    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "M/keep.txt")
        _add_file(tar, "M/skip.txt")

    archive = _archive(tmp_path, build)
    target = tmp_path / "out"
    target.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name != "M/skip.txt"]

    safe_extract(archive, path=target, members=members)

    assert (target / "M" / "keep.txt").exists()
    assert not (target / "M" / "skip.txt").exists()


def test_two_top_level_items_are_refused(tmp_path):
    def build(tar):
        _add_dir(tar, "A")
        _add_dir(tar, "B")

    with pytest.raises(ValueError, match="Expected one top-level folder"):
        safe_extract(_archive(tmp_path, build), path=tmp_path / "out")


def test_a_single_top_level_file_is_refused(tmp_path):
    def build(tar):
        _add_file(tar, "model.yaml")

    with pytest.raises(ValueError, match="Expected a directory"):
        safe_extract(_archive(tmp_path, build), path=tmp_path / "out")


def test_a_parent_directory_name_is_refused_and_nothing_is_written(tmp_path):
    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "M/../escaped.txt")

    target = tmp_path / "out"
    target.mkdir()

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(_archive(tmp_path, build), path=target)
    assert list(target.iterdir()) == []
    assert not (tmp_path / "escaped.txt").exists()


def test_an_absolute_name_is_refused(tmp_path):
    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "/M/abs.txt")

    target = tmp_path / "out"
    target.mkdir()

    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(_archive(tmp_path, build), path=target)
    assert list(target.iterdir()) == []
