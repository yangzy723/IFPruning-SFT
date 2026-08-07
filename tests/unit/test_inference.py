import tempfile
import unittest
from pathlib import Path

from inference import predictor_tokenizer_source


class PredictorTokenizerSourceTests(unittest.TestCase):
    def test_prefers_tokenizer_embedded_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedded = root / "checkpoint" / "predictor_tokenizer"
            embedded.mkdir(parents=True)
            fallback = root / "predictor"
            fallback.mkdir()
            self.assertEqual(
                predictor_tokenizer_source(embedded.parent, str(fallback)),
                str(embedded),
            )

    def test_uses_local_predictor_for_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            fallback = root / "predictor"
            fallback.mkdir()
            self.assertEqual(
                predictor_tokenizer_source(checkpoint, str(fallback)),
                str(fallback),
            )


if __name__ == "__main__":
    unittest.main()
