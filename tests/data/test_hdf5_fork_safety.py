"""HDF5 handles are not fork-safe: regression tests for the deadlock this caused.

Opening an ``h5py.File`` in the parent process before a ``fork``-started
DataLoader worker starts previously left every worker holding the same
(duplicated) file handle, which hangs on read rather than raising. These tests
exercise the real failure path end to end (under a hard timeout so a
regression fails the suite instead of hanging it), plus the individual pieces
of the fix: the cross-pid guard, the composed-dataset walk, and the
``preload=True`` handle close.
"""

import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from omniloader.data.datasets import HDF5Dataset
from omniloader.utils.seeding import _resettable_datasets, seed_worker
from tests.fixtures import write_hdf5


def _iterate_after_prefork_read(h5_path: str, result_queue: multiprocessing.Queue) -> None:
    """Child-process entry point: open+read in this process, then fork workers.

    Runs in its own process (spawned by the test) so that when it forks
    DataLoader workers, this process is the "parent" the workers inherit
    from, and a hang here cannot hang the test runner itself.
    """
    ds = HDF5Dataset(h5_path, "train")
    _ = ds[0]  # force the handle open here, before the DataLoader forks workers
    loader = DataLoader(
        ds,
        batch_size=2,
        num_workers=2,
        worker_init_fn=seed_worker,
        multiprocessing_context="fork",
    )
    batches = 0
    for _ in loader:
        batches += 1
    result_queue.put(batches)


class TestForkSafetyEndToEnd(unittest.TestCase):
    """Test 1 from the bug report: the failure needs training-loop-style use."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "data.h5")
        write_hdf5(Path(self.path), subset="train", n=10)

    def tearDown(self):
        self.tmp.cleanup()

    def test_iterating_after_prefork_read_completes_under_timeout(self):
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_iterate_after_prefork_read, args=(self.path, queue))
        proc.start()
        proc.join(timeout=30)
        still_running = proc.is_alive()
        if still_running:
            proc.terminate()
            proc.join()
        self.assertFalse(
            still_running, "DataLoader hung after a pre-fork read (fork safety regression)"
        )
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(queue.get(timeout=5), 5)  # 10 samples / batch_size 2


class TestHandleIdentityPerWorker(unittest.TestCase):
    """Test 2 from the bug report: each worker must open its own handle."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.h5"
        write_hdf5(self.path, subset="train", n=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reset_for_worker_clears_inherited_handle(self):
        ds = HDF5Dataset(self.path, "train")
        _ = ds[0]  # open the handle, simulating a pre-fork read
        inherited_handle = ds._file
        self.assertIsNotNone(inherited_handle)

        ds.reset_for_worker()

        self.assertIsNone(ds._file)
        self.assertEqual(len(ds._cache), 0)
        _ = ds[0]  # reopen, as a worker would on first use
        self.assertIsNot(ds._file, inherited_handle)
        self.assertTrue(inherited_handle)  # the stale handle is untouched, not closed by us


class _FakeComposed:
    """Minimal stand-in for OmniLoader's ``datasets`` composition shape."""

    def __init__(self, datasets):
        self.datasets = datasets


class TestComposedDatasetWalk(unittest.TestCase):
    """Test 3 from the bug report: a meta-dataset mixing corpora must be walked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path_a = Path(self.tmp.name) / "a.h5"
        self.path_b = Path(self.tmp.name) / "b.h5"
        write_hdf5(self.path_a, subset="train", n=3)
        write_hdf5(self.path_b, subset="train", n=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_both_leaves_through_composition(self):
        leaf_a = HDF5Dataset(self.path_a, "train")
        leaf_b = HDF5Dataset(self.path_b, "train")
        composed = _FakeComposed([leaf_a, leaf_b])

        found = _resettable_datasets(composed)

        self.assertEqual(set(map(id, found)), {id(leaf_a), id(leaf_b)})

    def test_nested_composition_is_deduplicated(self):
        leaf = HDF5Dataset(self.path_a, "train")
        inner = _FakeComposed([leaf])
        outer = _FakeComposed([inner, leaf])

        found = _resettable_datasets(outer)

        self.assertEqual(len(found), 1)
        self.assertIs(found[0], leaf)


class TestCrossPidGuard(unittest.TestCase):
    """Test 4 from the bug report: reading a foreign-pid handle must raise, not hang."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.h5"
        write_hdf5(self.path, subset="train", n=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_raises_when_handle_opened_in_another_pid(self):
        ds = HDF5Dataset(self.path, "train")
        _ = ds[0]  # opens the handle and records the current pid
        ds._pid = os.getpid() + 1  # simulate "opened elsewhere" without an actual fork

        with self.assertRaises(RuntimeError) as ctx:
            ds[1]
        self.assertIn("not fork-safe", str(ctx.exception))

    def test_same_pid_does_not_raise(self):
        ds = HDF5Dataset(self.path, "train")
        _ = ds[0]
        _ = ds[1]  # same process -> no guard trip


class TestPreloadClosesHandle(unittest.TestCase):
    """Test 5 from the bug report: preload=True must not leave a handle open."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data.h5"
        write_hdf5(self.path, subset="train", n=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_handle_closed_after_preload(self):
        ds = HDF5Dataset(self.path, "train", preload=True)
        self.assertIsNone(ds._file)
        self.assertIsNone(ds._pid)
        self.assertEqual(ds[0]["video"].shape, (16, 32))


if __name__ == "__main__":
    unittest.main()
