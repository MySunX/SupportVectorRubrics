from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rrd_rubricbench.data import load_rubricbench


class RubricBenchLoaderTest(unittest.TestCase):
    def test_load_rubricbench_normalizes_chosen_response_to_a(self) -> None:
        records = [
            {
                "case_id": "gold-a",
                "prompt_text": "prompt",
                "response_a": "chosen original A",
                "response_b": "rejected original B",
                "chosen_candidate": "a",
            },
            {
                "case_id": "gold-b",
                "prompt_text": "prompt",
                "response_a": "rejected original A",
                "response_b": "chosen original B",
                "chosen_candidate": "b",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rubricbench.json"
            path.write_text(json.dumps(records), encoding="utf-8")

            examples = load_rubricbench(path)

        self.assertEqual([example.gold_candidate for example in examples], ["a", "a"])
        self.assertEqual(examples[0].response_a, "chosen original A")
        self.assertEqual(examples[0].response_b, "rejected original B")
        self.assertEqual(examples[1].response_a, "chosen original B")
        self.assertEqual(examples[1].response_b, "rejected original A")


if __name__ == "__main__":
    unittest.main()

