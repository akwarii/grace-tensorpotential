"""Tests of ``tensorpotential/calculator/foundation_models.py``: the registry entries, the archive
extraction, the download with its retries, and the cache logic of ``get_or_download_*`` and ``grace_fm``.

Nothing here touches the network. ``requests.get`` is replaced by ``_FakeGet``, which serves bytes held in
memory (the only boundary mocked: the network); the archives are real ``tar.gz`` files written with ``tarfile``,
and the cache directory is a ``tmp_path`` given through the module attributes that hold it. The values compared
(URLs, names, file contents) are written out by hand, not computed with the unit under test.
``tests/test_origin_sidecar.py`` covers the provenance sidecar and ``tests/test_grace_models.py`` the command.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
import requests

from tensorpotential.calculator import foundation_models as fm

MODEL_NAME = "Toy-Model"
URL = "https://example.org/resolve/model-v9/models/Toy-Model-model.tar.gz"
CHECKPOINT_URL = (
    "https://example.org/resolve/checkpoint-v9/checkpoints/Toy-Model-checkpoint.tar.gz"
)
KOKKOS_URL = "https://example.org/resolve/kk-v9/kokkos/Toy-kokkos.npz"


# --- archives and a fake network ---


def _make_tar(path: Path, entries: dict[str, bytes | None]) -> bytes:
    """Write a ``.tar.gz`` of ``entries`` (a ``None`` content is a directory) and return its bytes."""
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return path.read_bytes()


def _model_archive(tmp_path: Path, top: str = "extracted") -> bytes:
    return _make_tar(
        tmp_path / "src.tar.gz", {top: None, f"{top}/data.txt": b"payload"}
    )


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        content_type: str = "application/gzip",
        with_length: bool = True,
    ):
        self._body = body
        self.headers = {"Content-Type": content_type}
        if with_length:
            self.headers["Content-Length"] = str(len(body))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, **_kwargs):
        for start in range(
            0, len(self._body), 4
        ):  # chunks smaller than asked for: the unit must keep them all
            yield self._body[start : start + 4]
        yield b""  # an empty keep-alive chunk is skipped


class _FakeGet:
    """Stands in for ``requests.get``: serves queued responses or raises queued exceptions, and records calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def network(monkeypatch):
    """Install a fake ``requests.get`` in the module; the test sets its outcomes."""

    def install(*outcomes):
        fake = _FakeGet(*outcomes)
        monkeypatch.setattr(fm.requests, "get", fake)
        return fake

    return install


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """The cache directories of the module in a temporary directory, with a toy model entry.

    Module state is test configuration here, not a replaced boundary: the cache location is read once at import,
    the registry is a plain dict, and ``_ORIGIN_WARNED`` is the once-per-process warning memory (reset so that
    a test sees its own warning). Every patch is undone by ``monkeypatch``.
    """
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(tmp_path / "cache" / "checkpoints")
    )
    monkeypatch.setitem(
        fm.MODELS_METADATA,
        MODEL_NAME,
        {
            fm.MODEL_URL_KEY: URL,
            fm.CHECKPOINT_URL_KEY: CHECKPOINT_URL,
            fm.KOKKOS_URL_KEY: KOKKOS_URL,
            fm.LICENSE_KEY: "Toy License",
        },
    )
    monkeypatch.setattr(fm, "_ORIGIN_WARNED", set())
    monkeypatch.delenv(fm.NO_UPDATE_CHECK_ENV, raising=False)
    return tmp_path / "cache"


# --- registry entries and names ---


def test_a_registry_entry_has_the_urls_of_its_release_tags_and_the_declared_capabilities():
    entry = fm._uqv6_entry("GRACE-X", "described", kokkos_base="GRACE-base")
    base = "https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/resolve"
    assert entry == {
        "url": f"{base}/model-v5-uq-v6/models/GRACE-X-model.tar.gz",
        "description": "described",
        "license": "Academic Software License",
        "checkpoint_url": f"{base}/checkpoint-v3-uq-v6/checkpoints/GRACE-X-checkpoint.tar.gz",
        "precision": "fp32",
        "uq": True,
        "parallel": True,
        "kokkos_url": f"{base}/kk-v3-uq-v6/kokkos/GRACE-base-kokkos.npz",
    }


def test_a_registry_entry_without_a_kokkos_base_has_no_kokkos_url_and_the_fp64_suffix_sets_the_precision():
    entry = fm._uqv6_entry("GRACE-X-fp64", "d", model_tag="m", checkpoint_tag="c")
    assert fm.KOKKOS_URL_KEY not in entry
    assert entry["precision"] == "fp64"
    assert entry["url"].endswith("/m/models/GRACE-X-fp64-model.tar.gz")
    assert entry["checkpoint_url"].endswith(
        "/c/checkpoints/GRACE-X-fp64-checkpoint.tar.gz"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GRACE-1L-OAM", "GRACE_1L_OAM"),
        ("a.b c", "a_b_c"),
        ("1abc", "_1abc"),
        ("_ok", "_ok"),
        ("class", "class_"),
        ("for-x", "for_x"),
        ("-lead", "_lead"),
    ],
)
def test_a_model_name_becomes_a_valid_identifier(name, expected):
    assert fm.to_valid_variable_name(name) == expected


def test_the_empty_name_has_no_identifier():
    with pytest.raises(
        IndexError
    ):  # pinned: the first character is read without a check
        fm.to_valid_variable_name("")


def test_every_registered_model_is_reachable_as_an_attribute_of_the_namespace_class():
    for name in fm.MODELS_NAME_LIST:
        assert getattr(fm.GRACEModels, fm.to_valid_variable_name(name)) == name
    assert (
        fm.MODELS_NAME_LIST[: len(fm.CORE_MODELS_NAME_LIST)] == fm.CORE_MODELS_NAME_LIST
    )


def test_every_alias_points_at_a_registered_model():
    for alias, target in fm.MODELS_ALIASES_DICT.items():
        assert target in fm.MODELS_METADATA, alias


# --- safe_extract ---


def test_a_single_top_level_folder_is_extracted_and_its_name_returned(tmp_path):
    archive = tmp_path / "ok.tar.gz"
    _make_tar(archive, {"top": None, "top/sub": None, "top/sub/f.txt": b"x"})
    out = tmp_path / "out"
    out.mkdir()
    assert fm.safe_extract(str(archive), path=str(out)) == "top"
    assert (out / "top" / "sub" / "f.txt").read_bytes() == b"x"


def test_a_top_level_file_instead_of_a_folder_is_refused(tmp_path):
    archive = tmp_path / "file.tar.gz"
    _make_tar(archive, {"lonely.txt": b"x"})
    with pytest.raises(
        ValueError, match=r"Expected a directory, but found a file: lonely\.txt"
    ):
        fm.safe_extract(str(archive), path=str(tmp_path))
    assert not (tmp_path / "lonely.txt").exists()


def test_two_top_level_items_are_refused_with_their_names(tmp_path):
    archive = tmp_path / "two.tar.gz"
    _make_tar(archive, {"a": None, "b": None})
    with pytest.raises(
        ValueError, match=r"Expected one top-level folder, found: \['a', 'b'\]"
    ):
        fm.safe_extract(str(archive), path=str(tmp_path))


def test_an_archive_with_nothing_in_it_is_refused(tmp_path):
    archive = tmp_path / "empty.tar.gz"
    _make_tar(archive, {})
    with pytest.raises(ValueError, match=r"found: \[\]"):
        fm.safe_extract(str(archive), path=str(tmp_path))


@pytest.mark.parametrize("bad", ["top/../escape.txt", "/top/abs.txt"])
def test_a_member_that_escapes_the_target_is_refused_and_nothing_is_extracted(
    tmp_path, bad
):
    archive = tmp_path / "bad.tar.gz"
    _make_tar(archive, {"top": None, "top/ok.txt": b"x", bad: b"y"})
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="Unsafe path detected"):
        fm.safe_extract(str(archive), path=str(out))
    assert list(out.iterdir()) == []
    assert not (tmp_path / "escape.txt").exists()


def test_an_absolute_top_level_member_counts_as_a_second_top_level_item(tmp_path):
    # pinned: the top-level check sees the empty first component of "/abs.txt" before the unsafe-path check
    archive = tmp_path / "abs.tar.gz"
    _make_tar(archive, {"top": None, "/abs.txt": b"y"})
    with pytest.raises(ValueError, match="Expected one top-level folder"):
        fm.safe_extract(str(archive), path=str(tmp_path))


# --- the streaming download ---


def test_a_download_writes_every_chunk_and_leaves_no_part_file(network, tmp_path):
    body = bytes(range(256)) * 3
    fake = network(_FakeResponse(body))
    dest = tmp_path / "file.bin"
    assert fm._stream_download("https://x/f", str(dest), desc="d") == str(dest)
    assert dest.read_bytes() == body
    assert not (tmp_path / "file.bin.part").exists()
    url, kwargs = fake.calls[0]
    assert url == "https://x/f"
    assert kwargs == {
        "stream": True,
        "timeout": fm.DOWNLOAD_TIMEOUT,
        "allow_redirects": True,
    }


def test_a_download_without_content_length_still_completes(network, tmp_path):
    network(_FakeResponse(b"abcdef", with_length=False))
    dest = tmp_path / "f"
    fm._stream_download("https://x/f", str(dest))
    assert dest.read_bytes() == b"abcdef"


def test_a_html_answer_is_refused_at_once_with_the_url_and_is_not_retried(
    network, tmp_path, capsys
):
    # the RuntimeError is not one of the transient errors that are retried
    fake = network(_FakeResponse(b"<html/>", content_type="text/html; charset=utf-8"))
    dest = tmp_path / "f"
    with pytest.raises(
        RuntimeError,
        match=r"Download failed \(got an HTML page, not a file\)\. Please check the URL 'https://x/f'",
    ):
        fm._stream_download("https://x/f", str(dest))
    assert len(fake.calls) == 1
    assert not dest.exists() and not (tmp_path / "f.part").exists()
    assert "retrying" not in capsys.readouterr().out


def test_a_persistent_network_error_is_reported_after_every_retry(
    network, tmp_path, capsys
):
    fake = network(requests.ConnectionError("reset"))
    with pytest.raises(
        RuntimeError, match=r"Download of 'https://x/f' failed after 3 attempts: reset"
    ):
        fm._stream_download("https://x/f", str(tmp_path / "f"))
    assert len(fake.calls) == fm.DOWNLOAD_RETRIES
    out = capsys.readouterr().out
    assert (
        "download attempt 1/3 failed (reset); retrying" in out
        and "download attempt 2/3 failed" in out
    )
    assert "download attempt 3/3" not in out


def test_a_transient_error_is_retried_and_the_next_attempt_succeeds(network, tmp_path):
    fake = network(
        requests.ConnectionError("reset"),
        requests.Timeout("slow"),
        _FakeResponse(b"final"),
    )
    dest = tmp_path / "f"
    fm._stream_download("https://x/f", str(dest))
    assert len(fake.calls) == 3
    assert dest.read_bytes() == b"final"


def test_a_half_written_part_file_is_removed_when_the_transfer_fails(network, tmp_path):
    class _Broken(_FakeResponse):
        def iter_content(self, **_kwargs):
            yield b"abc"
            raise requests.ConnectionError("cut")

    network(_Broken(b"abcdef"))
    dest = tmp_path / "f"
    with pytest.raises(RuntimeError, match="failed after 3 attempts: cut"):
        fm._stream_download("https://x/f", str(dest))
    assert not (tmp_path / "f.part").exists()
    assert not dest.exists()


def test_the_description_defaults_to_the_file_name(network, tmp_path, monkeypatch):
    seen = []

    class _Bar:
        def __init__(self, **kwargs):
            seen.append(kwargs["desc"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, _n):
            return None

    monkeypatch.setattr(fm, "tqdm", _Bar)  # the progress bar only draws on a terminal
    network(_FakeResponse(b"x"))
    fm._stream_download("https://x/f", str(tmp_path / "name.bin"))
    fm._stream_download("https://x/f", str(tmp_path / "name.bin"), desc="given")
    assert seen == ["name.bin", "given"]


# --- the package version ---


def test_the_package_version_is_a_string_and_unknown_when_the_metadata_is_missing(
    monkeypatch,
):
    import importlib.metadata as metadata

    assert isinstance(fm._tp_version(), str)

    def _missing(name):
        raise metadata.PackageNotFoundError(name)

    # what is replaced: the installed-package metadata, which is absent in a source checkout
    monkeypatch.setattr(metadata, "version", _missing)
    assert fm._tp_version() == "unknown"


# --- download, extract, rename ---


def test_an_archive_is_downloaded_unpacked_and_renamed_to_the_model_path(
    network, tmp_path
):
    network(_FakeResponse(_model_archive(tmp_path, top="extracted")))
    target = tmp_path / "cache" / "wanted"
    fm.download_extract_rename(URL, str(target) + "/")  # a trailing slash is ignored
    assert (target / "data.txt").read_bytes() == b"payload"
    assert not (tmp_path / "cache" / "extracted").exists()
    assert not (tmp_path / "cache" / fm.TMP_TAR_GZ).exists()


def test_an_archive_that_already_has_the_wanted_name_is_not_renamed(network, tmp_path):
    network(_FakeResponse(_model_archive(tmp_path, top="wanted")))
    target = tmp_path / "cache" / "wanted"
    fm.download_extract_rename(URL, str(target))
    assert (target / "data.txt").read_bytes() == b"payload"


@pytest.mark.parametrize(
    ("url", "requested"),
    [
        ("https://uni.sciebo.de/s/abc", "https://uni.sciebo.de/s/abc/download"),
        (
            "https://uni.sciebo.de/s/abc/download",
            "https://uni.sciebo.de/s/abc/download",
        ),
        (
            "https://huggingface.co/x/resolve/t/f.tar.gz",
            "https://huggingface.co/x/resolve/t/f.tar.gz",
        ),
    ],
)
def test_a_sciebo_link_gets_the_download_suffix_and_other_hosts_are_used_as_given(
    network, tmp_path, url, requested
):
    fake = network(_FakeResponse(_model_archive(tmp_path)))
    fm.download_extract_rename(url, str(tmp_path / "cache" / "m"))
    assert fake.calls[0][0] == requested


def test_an_unsafe_archive_is_not_unpacked_and_the_download_stays_behind(
    network, tmp_path
):
    bad = _make_tar(tmp_path / "bad.tar.gz", {"top": None, "top/../x": b"1"})
    network(_FakeResponse(bad))
    with pytest.raises(ValueError, match="Unsafe path"):
        fm.download_extract_rename(URL, str(tmp_path / "cache" / "m"))
    assert not (tmp_path / "cache" / "m").exists()


# --- get_or_download_model, checkpoint, kokkos ---


def test_a_missing_model_is_downloaded_stamped_with_its_origin_and_its_license_printed(
    cache, network, tmp_path, capsys
):
    network(_FakeResponse(_model_archive(tmp_path)))
    path = fm.get_or_download_model(MODEL_NAME)
    assert path == str(cache / MODEL_NAME)
    assert (cache / MODEL_NAME / "data.txt").read_bytes() == b"payload"
    record = json.loads((cache / MODEL_NAME / fm.ORIGIN_SIDECAR).read_text())
    assert (record["model_name"], record["payload"], record["url"], record["tag"]) == (
        MODEL_NAME,
        "model",
        URL,
        "model-v9",
    )
    assert record["kokkos_url"] == KOKKOS_URL
    out = capsys.readouterr().out
    assert "Downloading GRACE model" in out and "Model license: Toy License" in out


def test_a_cached_model_is_used_without_the_network_and_its_origin_is_checked(
    cache, network, capsys
):
    fake = network(requests.ConnectionError("must not be called"))
    (cache / MODEL_NAME).mkdir(parents=True)
    path = fm.get_or_download_model(MODEL_NAME)
    assert path == str(cache / MODEL_NAME)
    assert fake.calls == []
    captured = capsys.readouterr()
    assert "Using cached GRACE model" in captured.out
    assert "may be outdated" in captured.err  # no sidecar: counted as outdated


@pytest.mark.usefixtures("cache")
def test_an_explicit_model_path_is_used_and_never_checked_or_downloaded(
    network, monkeypatch, tmp_path, capsys
):
    local = tmp_path / "mine"
    local.mkdir()
    monkeypatch.setitem(fm.MODELS_METADATA[MODEL_NAME], fm.MODEL_PATH_KEY, str(local))
    network(requests.ConnectionError("must not be called"))
    assert fm.get_or_download_model(MODEL_NAME) == str(local)
    assert "may be outdated" not in capsys.readouterr().err


@pytest.mark.usefixtures("cache")
def test_a_model_without_url_that_is_not_cached_is_refused(monkeypatch):
    monkeypatch.delitem(fm.MODELS_METADATA[MODEL_NAME], fm.MODEL_URL_KEY)
    with pytest.raises(
        ValueError, match=f"No available URL to download for {MODEL_NAME}"
    ):
        fm.get_or_download_model(MODEL_NAME)


def test_an_aliased_name_resolves_to_the_registered_model(cache, monkeypatch):
    monkeypatch.setitem(fm.MODELS_ALIASES_DICT, "Old-Name", MODEL_NAME)
    (cache / MODEL_NAME).mkdir(parents=True)
    assert fm.get_or_download_model("Old-Name") == str(cache / MODEL_NAME)


def test_a_missing_license_is_reported_as_not_provided(cache, monkeypatch, capsys):
    monkeypatch.delitem(fm.MODELS_METADATA[MODEL_NAME], fm.LICENSE_KEY)
    (cache / MODEL_NAME).mkdir(parents=True)
    fm.get_or_download_model(MODEL_NAME)
    assert "Model license: not provided" in capsys.readouterr().out


def test_a_missing_checkpoint_is_downloaded_below_the_checkpoint_directory(
    cache, network, tmp_path, capsys
):
    network(_FakeResponse(_model_archive(tmp_path, top="ckpt")))
    path = fm.get_or_download_checkpoint(MODEL_NAME)
    assert path == str(cache / "checkpoints" / MODEL_NAME)
    assert (cache / "checkpoints" / MODEL_NAME / "data.txt").is_file()
    record = json.loads(
        (cache / "checkpoints" / MODEL_NAME / fm.ORIGIN_SIDECAR).read_text()
    )
    assert (record["payload"], record["url"], record["tag"]) == (
        "checkpoint",
        CHECKPOINT_URL,
        "checkpoint-v9",
    )
    assert "kokkos_url" not in record
    assert "GRACE model checkpoint downloaded to" in capsys.readouterr().out


def test_a_cached_checkpoint_is_checked_for_its_origin_unless_the_entry_names_a_path(
    cache, monkeypatch, tmp_path, capsys
):
    (cache / "checkpoints" / MODEL_NAME).mkdir(parents=True)
    assert fm.get_or_download_checkpoint(MODEL_NAME) == str(
        cache / "checkpoints" / MODEL_NAME
    )
    assert "may be outdated" in capsys.readouterr().err
    local = tmp_path / "mine"
    local.mkdir()
    monkeypatch.setitem(
        fm.MODELS_METADATA[MODEL_NAME], fm.CHECKPOINT_PATH_KEY, str(local)
    )
    assert fm.get_or_download_checkpoint(MODEL_NAME) == str(local)
    assert "may be outdated" not in capsys.readouterr().err


@pytest.mark.usefixtures("cache")
def test_a_checkpoint_without_url_that_is_not_cached_is_refused(monkeypatch):
    monkeypatch.delitem(fm.MODELS_METADATA[MODEL_NAME], fm.CHECKPOINT_URL_KEY)
    with pytest.raises(
        ValueError, match=f"No available checkpoint URL for {MODEL_NAME}"
    ):
        fm.get_or_download_checkpoint(MODEL_NAME)


def test_the_kokkos_file_is_downloaded_next_to_the_model_and_then_reused(
    cache, network, capsys
):
    network(_FakeResponse(b"npz-bytes"))
    path = fm.get_or_download_kokkos(MODEL_NAME)
    assert path == str(cache / MODEL_NAME / "kokkos.npz")
    assert Path(path).read_bytes() == b"npz-bytes"
    assert "kokkos.npz downloaded to" in capsys.readouterr().out
    network(requests.ConnectionError("must not be called"))
    assert fm.get_or_download_kokkos(MODEL_NAME) == path
    assert "Using cached kokkos.npz" in capsys.readouterr().out


@pytest.mark.usefixtures("cache")
def test_a_sciebo_kokkos_link_gets_the_download_suffix(network, monkeypatch):
    monkeypatch.setitem(
        fm.MODELS_METADATA[MODEL_NAME], fm.KOKKOS_URL_KEY, "https://uni.sciebo.de/s/k"
    )
    fake = network(_FakeResponse(b"x"))
    fm.get_or_download_kokkos(MODEL_NAME)
    assert fake.calls[0][0] == "https://uni.sciebo.de/s/k/download"


def test_a_kokkos_file_without_url_is_refused(cache, monkeypatch):
    monkeypatch.delitem(fm.MODELS_METADATA[MODEL_NAME], fm.KOKKOS_URL_KEY)
    with pytest.raises(
        ValueError, match=f"No available kokkos.npz URL for {MODEL_NAME}"
    ):
        fm.get_or_download_kokkos(MODEL_NAME)
    assert not (cache / MODEL_NAME).exists()


@pytest.mark.usefixtures("cache")
def test_the_kokkos_lookup_uses_an_explicit_model_path(monkeypatch, tmp_path):
    local = tmp_path / "mine"
    local.mkdir()
    (local / "kokkos.npz").write_bytes(b"")
    monkeypatch.setitem(fm.MODELS_METADATA[MODEL_NAME], fm.MODEL_PATH_KEY, str(local))
    assert fm.get_or_download_kokkos(MODEL_NAME) == str(local / "kokkos.npz")


# --- grace_fm ---


def test_an_unregistered_model_is_refused_with_the_list_of_names():
    with pytest.raises(AssertionError, match="model must be in"):
        fm.grace_fm("definitely-not-a-model")


def test_grace_fm_builds_a_calculator_on_the_model_path_with_the_given_padding(
    cache, monkeypatch
):
    from tensorpotential.calculator import asecalculator

    received = {}

    class _Calculator:
        def __init__(self, **kwargs):
            received.update(kwargs)

    # what is replaced: the calculator, whose construction needs a saved model (13 s) and is tested in
    # test_calculator.py; the model lookup is real, on a cached directory, so no download is attempted
    monkeypatch.setattr(asecalculator, "TPCalculator", _Calculator)
    (cache / MODEL_NAME).mkdir(parents=True)
    monkeypatch.setattr(fm, "MODELS_NAME_LIST", [*fm.MODELS_NAME_LIST, MODEL_NAME])
    calc = fm.grace_fm(MODEL_NAME, pad_atoms_number=3, min_dist=0.5, extra_option="e")
    assert isinstance(calc, _Calculator)
    assert received == {
        "model": str(cache / MODEL_NAME),
        "pad_neighbors_fraction": 0.05,
        "pad_atoms_number": 3,
        "max_number_reduction_recompilation": 2,
        "min_dist": 0.5,
        "extra_option": "e",
    }


def test_the_docstrings_name_every_registered_model():
    for name in fm.CORE_MODELS_NAME_LIST:
        assert name in (fm.get_or_download_model.__doc__ or "")
        assert name in (fm.grace_fm.__doc__ or "")


def test_the_module_registers_models_from_the_user_registry_without_overriding(
    tmp_path,
):
    """The registry file is read at import: a fresh interpreter sees a user model and keeps a built-in untouched."""
    import subprocess
    import sys

    cache_dir = tmp_path / "grace"
    cache_dir.mkdir()
    (cache_dir / fm.MODELS_REGISTRY_YAML).write_text(
        "UserModel:\n  path: /somewhere\n  license: mine\nGRACE-2L-OAM:\n  path: /should/not/win\n"
    )
    code = (
        "import tensorpotential;"
        "from tensorpotential.calculator import foundation_models as fm;"
        "print(fm.MODELS_METADATA['UserModel']['path'], fm.MODELS_METADATA['GRACE-2L-OAM'].get('path'),"
        " 'UserModel' in fm.MODELS_NAME_LIST, fm.FOUNDATION_CACHE_DIR == r'"
        + str(cache_dir)
        + "')"
    )
    repository = (
        Path(fm.__file__).resolve().parents[2]
    )  # the tree under test, whatever the working directory
    env = {**os.environ, fm.GRACE_CACHE: str(cache_dir), "PYTHONPATH": str(repository)}
    result = subprocess.run(  # noqa: S603  (the interpreter and a literal program, no outside input)
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split()[-4:] == ["/somewhere", "None", "True", "True"]
