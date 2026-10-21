import sys
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_direct_python_guidance_respects_minimal_mode(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            with (patch.object(bootstrap, "VENV_DIRECTORY", Path(directory)),
                  patch.object(bootstrap, "_venv_python", return_value=Path(sys.executable)),
                  patch.object(bootstrap, "_run")):
                bootstrap.main(["--minimal", "--skip-data"])
        self.assertIn("Activation is not required", output.getvalue())
        self.assertIn("-m cifar10_lab train --model perceptron --quick --device cpu", output.getvalue())
        self.assertNotIn("-m jupyter lab", output.getvalue())
        if sys.platform == "win32":
            self.assertIn(f'& "{sys.executable}"', output.getvalue())

    def test_minimal_setup_skips_notebook_extras_and_data_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(bootstrap, "VENV_DIRECTORY", Path(temporary_directory)),
                patch.object(bootstrap, "_venv_python", return_value=Path(sys.executable)),
                patch.object(bootstrap, "_run") as run,
            ):
                result = bootstrap.main(["--minimal", "--skip-data"])

        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[1].args[0][-1], ".")
        self.assertEqual(
            run.call_args_list[2].args[0][-2:],
            ["cifar10_lab", "doctor"],
        )


if __name__ == "__main__":
    unittest.main()
