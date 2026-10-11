"""Tests of ``calculator/foundation_models.py``: the extraction of a downloaded model archive.

A foundation-model archive comes from the network, so ``safe_extract`` is a trust boundary. Logic layer only (no
physics here): what it accepts, what it refuses (names, links, special files), and where the files land. Archives are built in ``tmp_path``.
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


def _add_link(
    tar: tarfile.TarFile, name: str, target: str, kind: bytes = tarfile.SYMTYPE
) -> None:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
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


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        (tarfile.SYMTYPE, "../../outside"),  # relative link out of the target directory
        (tarfile.SYMTYPE, "/etc"),  # absolute link
        (tarfile.LNKTYPE, "../outside"),  # hard link out of the target directory
    ],
)
def test_a_link_that_escapes_the_target_is_refused(tmp_path, kind, target):
    def build(tar):
        _add_dir(tar, "M")
        _add_link(tar, "M/link", target, kind)

    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract(_archive(tmp_path, build), path=out)
    assert not (out / "M" / "link").is_symlink()


def test_a_file_written_through_a_planted_link_is_refused(tmp_path):
    """A safe-looking link followed by a regular member below it: the write would land outside ``out``."""
    victim = tmp_path / "victim"
    victim.mkdir()

    def build(tar):
        _add_dir(tar, "M")
        _add_link(tar, "M/dir", str(victim))
        _add_file(tar, "M/dir/payload.txt", b"owned")

    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract(_archive(tmp_path, build), path=out)
    assert list(victim.iterdir()) == []


def test_a_link_that_stays_inside_the_target_is_extracted(tmp_path):
    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "M/real.txt", b"data")
        _add_link(tar, "M/alias.txt", "real.txt")

    out = tmp_path / "out"
    out.mkdir()

    safe_extract(_archive(tmp_path, build), path=out)

    assert (out / "M" / "alias.txt").read_bytes() == b"data"


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_without_the_data_filter_every_link_and_special_file_is_refused(
    tmp_path, monkeypatch, kind
):
    # Replaces the capability of the interpreter (the filter exists from Python 3.11.4), the one thing a
    # test cannot choose; the member check that stands in for it is the unit under test.
    monkeypatch.delattr(tarfile, "data_filter")

    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "M/real.txt")
        _add_link(tar, "M/odd", "real.txt", kind)

    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="link or special file"):
        safe_extract(_archive(tmp_path, build), path=out)
    assert list(out.iterdir()) == []


def test_without_the_data_filter_a_plain_archive_is_extracted(tmp_path, monkeypatch):
    monkeypatch.delattr(tarfile, "data_filter")  # see above

    def build(tar):
        _add_dir(tar, "M")
        _add_file(tar, "M/model.yaml", b"a: 1\n")

    out = tmp_path / "out"
    out.mkdir()

    assert safe_extract(_archive(tmp_path, build), path=out) == "M"
    assert (out / "M" / "model.yaml").read_bytes() == b"a: 1\n"
