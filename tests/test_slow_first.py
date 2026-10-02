from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .slow_first import deal_slow_first, first_block_size

REPO_ROOT = Path(__file__).resolve().parent.parent


def _slow(name: str) -> bool:
    return name.startswith("slow")


def _tests(n_fast: int, n_slow: int) -> list[str]:
    # slow tests at the end of the collection, as they sit alphabetically in places
    return [*(f"fast{i}" for i in range(n_fast)), *(f"slow{i}" for i in range(n_slow))]


def test_first_block_size_is_a_quarter_of_a_workers_share_and_at_least_two() -> None:
    # pytest-xdist LoadScheduling: max((n // workers) // 4, 2), by hand
    assert first_block_size(100, 4) == 6  # 25 per worker, a quarter
    assert first_block_size(1287, 4) == 80  # 321 per worker
    assert first_block_size(1287, 1) == 321
    assert first_block_size(10, 4) == 2  # 2 per worker, floor of two
    assert first_block_size(0, 3) == 2


def test_each_workers_first_block_starts_with_its_share_of_slow_tests() -> None:
    items = _tests(40, 5)
    ordered = deal_slow_first(items, _slow, workers=3)
    chunk = first_block_size(len(items), 3)  # 3
    assert [ordered[k * chunk] for k in range(3)] == ["slow0", "slow1", "slow2"]
    # worker 0 also gets slow3 and worker 1 slow4, right after their heads
    assert ordered[1] == "slow3"
    assert ordered[chunk + 1] == "slow4"


def test_every_test_is_kept_once_and_the_others_keep_their_order() -> None:
    items = _tests(40, 5)
    ordered = deal_slow_first(items, _slow, workers=4)
    assert sorted(ordered) == sorted(items)
    assert [i for i in ordered if not _slow(i)] == [i for i in items if not _slow(i)]


def test_the_order_is_unchanged_without_workers_or_without_slow_tests() -> None:
    items = _tests(10, 3)
    assert deal_slow_first(items, _slow, workers=1) == items
    assert deal_slow_first(items, _slow, workers=0) == items
    plain = _tests(10, 0)
    assert deal_slow_first(plain, _slow, workers=4) == plain


def test_a_block_of_slow_tests_longer_than_the_first_block_is_kept_whole() -> None:
    items = _tests(3, 5)  # first block = max(8 // 4 // 2, 2) = 2; worker 0 gets 3 slow
    ordered = deal_slow_first(items, _slow, workers=2)
    assert ordered[:3] == ["slow0", "slow2", "slow4"]
    assert sorted(ordered) == sorted(items)


def test_the_hook_reorders_in_every_xdist_worker_and_not_in_a_serial_run(
    tmp_path: Path,
) -> None:
    # A real pytest session with two xdist workers; each worker writes the collection
    # order it ended up with.
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = slow: slow tests\n")
    (tmp_path / "conftest.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from tests.slow_first import deal_slow_first\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    workers = getattr(config, 'workerinput', {}).get('workercount', 1)\n"
        "    items[:] = deal_slow_first(\n"
        "        items, lambda it: it.get_closest_marker('slow') is not None, workers)\n"
        "    who = os.environ.get('PYTEST_XDIST_WORKER', 'serial')\n"
        "    Path(f'order_{who}.txt').write_text('\\n'.join(it.name for it in items))\n"
    )
    body = ["import pytest"]
    body += [f"def test_fast{i}(): ..." for i in range(12)]
    body += [f"@pytest.mark.slow\ndef test_slow{i}(): ..." for i in range(2)]
    (tmp_path / "test_x.py").write_text("\n".join(body) + "\n")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))

    def run(*args: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run("-n", "2")
    chunk = first_block_size(14, 2)  # 2
    for worker in ("gw0", "gw1"):
        order = (tmp_path / f"order_{worker}.txt").read_text().split("\n")
        assert order[0] == "test_slow0"
        assert order[chunk] == "test_slow1"
        assert len(order) == 14
    run()
    serial = (tmp_path / "order_serial.txt").read_text().split("\n")
    assert serial[-2:] == ["test_slow0", "test_slow1"]
