from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_rrd_rubricbench import _run_checkpointed_stage


class CheckpointingTest(unittest.TestCase):
    def test_checkpointed_stage_reuses_completed_items(self) -> None:
        calls: list[int] = []

        def compute(idx: int) -> dict[str, int]:
            calls.append(idx)
            return {"value": idx * 10}

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir) / "stage"
            examples = ["a", "b", "c"]
            keys = ["000_a", "001_b", "002_c"]

            first = _run_checkpointed_stage(
                stage_name="test",
                examples=examples,
                keys=keys,
                stage_dir=stage_dir,
                max_workers=2,
                compute=compute,
                dump=lambda item: item,
                load=lambda item: item,
            )
            self.assertEqual(first, [{"value": 0}, {"value": 10}, {"value": 20}])
            self.assertEqual(sorted(calls), [0, 1, 2])

            calls.clear()
            second = _run_checkpointed_stage(
                stage_name="test",
                examples=examples,
                keys=keys,
                stage_dir=stage_dir,
                max_workers=2,
                compute=compute,
                dump=lambda item: item,
                load=lambda item: item,
            )

        self.assertEqual(second, first)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

