import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import analysis


class TestPipeline(unittest.TestCase):
    def test_steps_with_and_without_pca(self):
        self.assertEqual(list(analysis.build_pipeline(10).named_steps), ["scaler", "pca", "svm"])
        self.assertEqual(analysis.build_pipeline(10).named_steps["pca"].n_components, 10)
        self.assertEqual(list(analysis.build_pipeline().named_steps), ["scaler", "svm"])


class TestMain(unittest.TestCase):
    def test_model_selection_uses_training_set_only_and_saves_figures(self):
        seen_sizes = []
        original = analysis.cross_validate_components

        def recording(X, y, options):
            seen_sizes.append(len(X))
            return original(X, y, options)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(analysis, "cross_validate_components", recording), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            analysis.main(Path(tmp))
            saved = sorted(p.name for p in Path(tmp).iterdir())

        self.assertEqual(set(seen_sizes), {1437})
        self.assertEqual(saved, ["confusion_matrix.png", "cv_accuracy.png",
                                 "explained_variance.png", "pca_projection.png"])
        self.assertIn("Selected number of components: 50", output.getvalue())


if __name__ == "__main__":
    unittest.main()
