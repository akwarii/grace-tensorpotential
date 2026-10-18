"""The typed errors: each is a ``TensorPotentialError`` and the builtin type it replaces, and is raised where it was."""

from __future__ import annotations

import io
import tarfile

import pytest

from tensorpotential import errors
from tensorpotential.calculator.foundation_models import safe_extract
from tensorpotential.core import backends
from tensorpotential.instructions.base import str_to_class

BUILTIN_PARENT = {
    "ModelFormatError": ValueError,
    "ClassPathError": ValueError,
    "ArchiveError": ValueError,
    "NeighbourListError": ValueError,
    "KerasVariableUnavailableError": RuntimeError,
}


@pytest.mark.parametrize(("name", "builtin"), sorted(BUILTIN_PARENT.items()))
def test_each_error_is_the_package_base_and_the_builtin_it_replaces(name, builtin):
    error = getattr(errors, name)
    assert issubclass(error, errors.TensorPotentialError)
    assert issubclass(error, builtin)


def test_the_backend_error_is_a_package_error_and_an_import_error():
    error = backends.BackendNotInstalledError("tf", ("tensorflow",), "needs tensorflow")
    assert isinstance(error, errors.TensorPotentialError)
    assert isinstance(error, ImportError)
    assert error.name == "tensorflow"
    assert error.backend == "tf"
    assert str(error) == "needs tensorflow"


def test_the_moved_classes_keep_their_old_import_paths():
    from tensorpotential.keras_adapter import KerasVariableUnavailableError
    from tensorpotential.utils import ModelFormatError

    assert ModelFormatError is errors.ModelFormatError
    assert KerasVariableUnavailableError is errors.KerasVariableUnavailableError


def test_a_malformed_class_path_raises_class_path_error():
    with pytest.raises(
        errors.ClassPathError, match="Couldn't deserialize class `not a path`"
    ):
        str_to_class("not a path")
    with pytest.raises(errors.ClassPathError, match="has no attribute"):
        str_to_class("os.no_such_name")


def test_a_class_path_to_the_wrong_kind_of_object_raises_class_path_error():
    with pytest.raises(errors.ClassPathError, match="not a subclass of"):
        str_to_class("os.path.join", base=Exception)


def _archive(tmp_path, members):
    path = tmp_path / "a.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                data = b"x"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_an_archive_with_two_top_level_folders_raises_archive_error(tmp_path):
    archive = _archive(tmp_path, [("a", "dir"), ("b", "dir")])
    with pytest.raises(errors.ArchiveError, match="Expected one top-level folder"):
        safe_extract(archive, str(tmp_path / "out"))


def test_an_archive_with_a_top_level_file_raises_archive_error(tmp_path):
    archive = _archive(tmp_path, [("a", "file")])
    with pytest.raises(errors.ArchiveError, match="Expected a directory"):
        safe_extract(archive, str(tmp_path / "out"))


def test_an_archive_member_outside_the_folder_raises_archive_error(tmp_path):
    archive = _archive(tmp_path, [("top", "dir"), ("top/../evil", "file")])
    with pytest.raises(errors.ArchiveError, match="Unsafe path detected"):
        safe_extract(archive, str(tmp_path / "out"))
