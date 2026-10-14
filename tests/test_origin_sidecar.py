"""Provenance sidecar: a cached payload must be recognised as stale when the
installed package points at a different release tag.

See pgs/uq/deployment_protocol.md §7 — the cache is keyed by model name only, so
without this the v5 re-export would never reach anyone who already had a model.
"""

import pytest

from tensorpotential.calculator import foundation_models as fm

V3 = "https://huggingface.co/x/y/resolve/model-v3-uq-v6/models/M-model.tar.gz"
V5 = "https://huggingface.co/x/y/resolve/model-v5-uq-v6/models/M-model.tar.gz"

pytestmark = pytest.mark.usefixtures("info_logs")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    fm._ORIGIN_WARNED.clear()
    monkeypatch.delenv(fm.NO_UPDATE_CHECK_ENV, raising=False)


def test_tag_is_parsed_from_resolve_url():
    assert fm._tag_from_url(V5) == "model-v5-uq-v6"
    assert fm._tag_from_url("https://example.org/no/tag/here.tar.gz") is None


def test_roundtrip_records_url_and_tag(tmp_path):
    fm.write_origin_sidecar(str(tmp_path), "M", "model", V5)
    rec = fm.read_origin_sidecar(str(tmp_path))
    assert rec["url"] == V5
    assert rec["tag"] == "model-v5-uq-v6"
    assert rec["model_name"] == "M" and rec["payload"] == "model"


def test_matching_url_is_current(tmp_path, caplog):
    fm.write_origin_sidecar(str(tmp_path), "M", "model", V5)
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is True
    assert caplog.text == ""


def test_stale_tag_warns(tmp_path, caplog):
    fm.write_origin_sidecar(str(tmp_path), "M", "model", V3)
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is False
    err = caplog.text
    assert "model-v3-uq-v6" in err and "model-v5-uq-v6" in err
    assert "grace_models update M" in err


def test_missing_sidecar_counts_as_outdated(tmp_path, caplog):
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is False
    assert "no provenance record" in caplog.text


def test_entry_without_url_is_never_checked(tmp_path, caplog):
    """Local-path, user-registry and experimental entries have no URL to compare."""
    assert fm.check_origin(str(tmp_path), "M", "model", None) is True
    assert caplog.text == ""


def test_env_var_suppresses(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(fm.NO_UPDATE_CHECK_ENV, "1")
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is True
    assert caplog.text == ""


def test_warns_only_once_per_model_and_payload(tmp_path, caplog):
    """TPCalculator can be constructed in a loop; the warning must not spam."""
    for _ in range(3):
        fm.check_origin(str(tmp_path), "M", "model", V5)
    assert caplog.text.count("may be outdated") == 1
    caplog.clear()
    fm.check_origin(str(tmp_path), "M", "checkpoint", V5)
    assert caplog.text.count("may be outdated") == 1


def test_unwritable_dir_does_not_raise(tmp_path, monkeypatch):
    """A read-only cache must never break a download.

    Raise from `open` rather than relying on `chmod 0o500`: root ignores the mode,
    so under a root CI container the permission trick would exercise nothing and
    then fail the assertion.
    """
    import builtins

    real_open = builtins.open

    def refuse(path, *a, **k):
        if str(path).endswith(fm.ORIGIN_SIDECAR):
            raise OSError(30, "Read-only file system")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", refuse)
    fm.write_origin_sidecar(str(tmp_path), "M", "model", V5)  # must not raise
    assert fm.read_origin_sidecar(str(tmp_path)) is None


def test_corrupt_sidecar_is_treated_as_missing(tmp_path, capsys):
    (tmp_path / fm.ORIGIN_SIDECAR).write_text("{not json")
    assert fm.read_origin_sidecar(str(tmp_path)) is None
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is False


def test_shipped_models_point_at_v5(tmp_path):
    """Every uqv6 entry must serve its SavedModel from the v5 tag; checkpoints stay."""
    for name in fm._UQV6_MODEL_NAMES + fm._UQV6_V4_MODEL_NAMES:
        e = fm.MODELS_METADATA[name]
        assert f"/resolve/{fm.UQV6_V5_MODEL_TAG}/" in e[fm.MODEL_URL_KEY], name
        assert fm.UQV6_V5_MODEL_TAG not in e[fm.CHECKPOINT_URL_KEY], name
    for name in fm._UQV6_MODEL_NAMES:
        e = fm.MODELS_METADATA[f"{name}-fp64"]
        assert f"/resolve/{fm.UQV6_V5_MODEL_TAG}/" in e[fm.MODEL_URL_KEY], name


# --- review follow-ups (MR !57) --------------------------------------------


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "  "])
def test_env_var_negative_spellings_keep_the_check_on(tmp_path, monkeypatch, caplog, val):
    """A bare truthiness test on the raw string would fail open on exactly these."""
    monkeypatch.setenv(fm.NO_UPDATE_CHECK_ENV, val)
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is False
    assert "may be outdated" in caplog.text


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_env_var_positive_spellings_suppress(tmp_path, monkeypatch, caplog, val):
    monkeypatch.setenv(fm.NO_UPDATE_CHECK_ENV, val)
    assert fm.check_origin(str(tmp_path), "M", "model", V5) is True
    assert caplog.text == ""


def test_checkpoint_getter_resolves_aliases(monkeypatch):
    """get_or_download_model remapped aliases; the checkpoint getter did not,
    so an aliased name raised KeyError instead of resolving."""
    alias, canonical = next(
        (a, c) for a, c in fm.MODELS_ALIASES_DICT.items() if c in fm.MODELS_METADATA
    )
    seen = {}

    def fake_isdir(p):
        seen["path"] = p
        return True

    monkeypatch.setattr(fm.os.path, "isdir", fake_isdir)
    monkeypatch.setattr(fm, "check_origin", lambda *a, **k: True)
    fm.get_or_download_checkpoint(alias)  # must not raise KeyError
    assert canonical in seen["path"]


def test_local_path_entries_are_not_checked(tmp_path, caplog, monkeypatch):
    """An entry with an explicit `path` points at a directory the user curates."""
    name = "_local_only_model"
    monkeypatch.setitem(
        fm.MODELS_METADATA, name,
        {fm.MODEL_PATH_KEY: str(tmp_path), fm.MODEL_URL_KEY: V5},
    )
    fm.get_or_download_model(name)
    assert "may be outdated" not in caplog.text


def _update_args(**kw):
    import argparse

    return argparse.Namespace(
        model_name=kw.get("model_name", []), all=kw.get("all", False),
        force=kw.get("force", False), dry_run=kw.get("dry_run", False),
    )


def _fake_cached_model(tmp_path, monkeypatch, name="_fake_model", url=V3):
    """A cache dir holding a payload plus a side-loaded file, stamped as stale."""
    d = tmp_path / name
    d.mkdir()
    (d / "saved_model.pb").write_text("payload")
    (d / "kokkos.npz").write_text("side-loaded")
    fm.write_origin_sidecar(str(d), name, "model", url)
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setitem(fm.MODELS_METADATA, name, {fm.MODEL_URL_KEY: V5})
    return d


def test_failed_update_restores_the_previous_payload(tmp_path, monkeypatch, capsys):
    """rmtree-then-fetch destroyed the cache when the download failed."""
    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch)

    def boom(_name):
        raise RuntimeError("network died")

    monkeypatch.setattr(fm, "get_or_download_model", boom)
    update_models(_update_args(model_name=["_fake_model"]))

    assert (d / "saved_model.pb").read_text() == "payload", "payload was destroyed"
    assert (d / "kokkos.npz").read_text() == "side-loaded"
    assert not (tmp_path / "_fake_model.pre-update").exists(), "stash left behind"
    assert "restored" in capsys.readouterr().out


def test_successful_update_swaps_in_the_new_payload(tmp_path, monkeypatch):
    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch)

    def fake_fetch(name):
        new = tmp_path / name
        new.mkdir()
        (new / "saved_model.pb").write_text("new payload")
        fm.write_origin_sidecar(str(new), name, "model", V5)

    monkeypatch.setattr(fm, "get_or_download_model", fake_fetch)
    update_models(_update_args(model_name=["_fake_model"]))

    assert (d / "saved_model.pb").read_text() == "new payload"
    assert fm.read_origin_sidecar(str(d))["tag"] == "model-v5-uq-v6"
    assert not (tmp_path / "_fake_model.pre-update").exists()


def test_update_refuses_an_explicit_local_path(tmp_path, monkeypatch, capsys):
    """An entry with `path` points at a user-curated dir; rmtree there is not ours."""
    from tensorpotential.scripts.grace_models import update_models

    d = tmp_path / "curated"
    d.mkdir()
    (d / "saved_model.pb").write_text("mine")
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setitem(
        fm.MODELS_METADATA, "_pathy",
        {fm.MODEL_PATH_KEY: str(d), fm.MODEL_URL_KEY: V5},
    )
    monkeypatch.setattr(
        fm, "get_or_download_model",
        lambda n: pytest.fail("must not download over a curated path"),
    )
    update_models(_update_args(model_name=["_pathy"]))
    assert (d / "saved_model.pb").read_text() == "mine"
    assert "explicit local path" in capsys.readouterr().out


# --- update_models: the cached-model walk (characterization) ----------------------------------------------------------


def _no_download(monkeypatch):
    for fetcher in ("get_or_download_model", "get_or_download_checkpoint"):
        monkeypatch.setattr(
            fm, fetcher, lambda n: pytest.fail("nothing may be downloaded")
        )


def _summary(out: str) -> str:
    return out.strip().splitlines()[-1]


def test_update_without_a_model_name_does_nothing(capsys):
    from tensorpotential.scripts.grace_models import update_models

    update_models(_update_args())
    assert capsys.readouterr().out == (
        "Nothing to do: give one or more model names, or --all.\n"
    )


def test_update_skips_an_unknown_model_and_counts_it(monkeypatch, capsys):
    from tensorpotential.scripts.grace_models import update_models

    _no_download(monkeypatch)
    update_models(_update_args(model_name=["_not_registered"]))
    out = capsys.readouterr().out
    assert "  _not_registered: not a known model - skipped\n" in out
    assert _summary(out) == "up-to-date: 0   updated: 0   skipped: 1"


def test_update_all_walks_the_core_models(tmp_path, monkeypatch, capsys):
    from tensorpotential.scripts.grace_models import update_models

    _fake_cached_model(tmp_path, monkeypatch)
    monkeypatch.setattr(fm, "CORE_MODELS_NAME_LIST", ["_fake_model"])
    update_models(_update_args(all=True, dry_run=True))
    assert "  _fake_model [model]: model-v3-uq-v6 -> model-v5-uq-v6\n" in (
        capsys.readouterr().out
    )


def test_a_current_payload_is_counted_and_left_alone_unless_forced(
    tmp_path, monkeypatch, capsys
):
    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch, url=V5)
    _no_download(monkeypatch)
    update_models(_update_args(model_name=["_fake_model"]))
    out = capsys.readouterr().out
    assert " -> " not in out
    assert _summary(out) == "up-to-date: 1   updated: 0   skipped: 0"
    assert (d / "saved_model.pb").read_text() == "payload"

    update_models(_update_args(model_name=["_fake_model"], force=True, dry_run=True))
    out = capsys.readouterr().out
    assert "  _fake_model [model]: model-v5-uq-v6 -> model-v5-uq-v6\n" in out
    assert _summary(out) == "up-to-date: 0   would update: 1   skipped: 0"


@pytest.mark.parametrize(
    ("record", "was"),
    [(False, "no provenance record"), (True, "?")],
    ids=["no-sidecar", "sidecar-without-tag"],
)
def test_the_printed_origin_of_a_payload_without_a_tag(
    tmp_path, monkeypatch, capsys, record, was
):
    """A payload with no sidecar, or a sidecar that holds no tag, is reported and its new URL printed untagged."""
    import json

    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch)
    sidecar = d / fm.ORIGIN_SIDECAR
    if record:
        sidecar.write_text(json.dumps({"url": V3}))
    else:
        sidecar.unlink()
    plain = "https://example.org/no/tag/here.tar.gz"
    monkeypatch.setitem(fm.MODELS_METADATA, "_fake_model", {fm.MODEL_URL_KEY: plain})
    update_models(_update_args(model_name=["_fake_model"], dry_run=True))
    assert f"  _fake_model [model]: {was} -> {plain}\n" in capsys.readouterr().out


def test_a_symlinked_cache_entry_is_refused(tmp_path, monkeypatch, capsys):
    from tensorpotential.scripts.grace_models import update_models

    store = tmp_path / "store"
    store.mkdir()
    (store / "saved_model.pb").write_text("curated")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "_linked").symlink_to(store, target_is_directory=True)
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(cache))
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setitem(fm.MODELS_METADATA, "_linked", {fm.MODEL_URL_KEY: V5})
    _no_download(monkeypatch)
    update_models(_update_args(model_name=["_linked"]))
    out = capsys.readouterr().out
    assert f"REFUSING: {cache / '_linked'} is a symlink -> {store}" in out
    assert _summary(out) == "up-to-date: 0   updated: 0   skipped: 1"
    assert (store / "saved_model.pb").read_text() == "curated"


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch)
    _no_download(monkeypatch)
    update_models(_update_args(model_name=["_fake_model"], dry_run=True))
    out = capsys.readouterr().out
    assert _summary(out) == "up-to-date: 0   would update: 1   skipped: 0"
    assert (d / "saved_model.pb").read_text() == "payload"
    assert not (tmp_path / "_fake_model.pre-update").exists()


def test_an_update_replaces_a_stash_left_by_an_earlier_run_and_notes_the_kokkos_file(
    tmp_path, monkeypatch, capsys
):
    from tensorpotential.scripts.grace_models import update_models

    d = _fake_cached_model(tmp_path, monkeypatch)
    stale = tmp_path / "_fake_model.pre-update"
    stale.mkdir()
    (stale / "old").write_text("leftover")

    def fake_fetch(name):
        (tmp_path / name).mkdir()

    monkeypatch.setattr(fm, "get_or_download_model", fake_fetch)
    update_models(_update_args(model_name=["_fake_model"]))
    out = capsys.readouterr().out
    assert not stale.exists()
    assert not (d / "kokkos.npz").exists()
    assert "note: kokkos.npz was replaced; re-fetch with `grace_models download _fake_model --kokkos`" in out
    assert _summary(out) == "up-to-date: 0   updated: 1   skipped: 0"


def test_a_cached_checkpoint_is_updated_through_the_checkpoint_fetcher(
    tmp_path, monkeypatch, capsys
):
    from tensorpotential.scripts.grace_models import update_models

    ckpt_root = tmp_path / "ckpt"
    d = ckpt_root / "_fake_model"
    d.mkdir(parents=True)
    (d / "checkpoint.index").write_text("old")
    fm.write_origin_sidecar(str(d), "_fake_model", "checkpoint", V3)
    monkeypatch.setattr(fm, "FOUNDATION_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(fm, "FOUNDATION_CHECKPOINTS_CACHE_DIR", str(ckpt_root))
    monkeypatch.setitem(
        fm.MODELS_METADATA, "_fake_model", {fm.CHECKPOINT_URL_KEY: V5}
    )
    fetched = []

    def fake_fetch(name):
        fetched.append(name)
        (ckpt_root / name).mkdir()

    monkeypatch.setattr(fm, "get_or_download_checkpoint", fake_fetch)
    monkeypatch.setattr(
        fm, "get_or_download_model", lambda n: pytest.fail("a checkpoint only")
    )
    update_models(_update_args(model_name=["_fake_model"]))
    out = capsys.readouterr().out
    assert fetched == ["_fake_model"]
    assert "  _fake_model [checkpoint]: model-v3-uq-v6 -> model-v5-uq-v6\n" in out
    assert "kokkos" not in out


def test_the_public_tag_parser_is_the_function_the_private_name_resolves_to():
    assert fm.tag_from_url is fm._tag_from_url
    assert fm.tag_from_url(V3) == "model-v3-uq-v6"
