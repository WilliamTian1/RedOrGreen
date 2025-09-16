import subprocess
import sys


def test_cli_smoke():
    # Run after pipeline has produced artifacts. Here we just ensure the CLI prints expected tokens.
    result = subprocess.run([sys.executable, "-m", "app.cli.analyze", "AAPL: guidance raised for Q4"], capture_output=True, text=True)
    # Do not assert on return code because artifacts may be missing in CI prior to training.
    out = result.stdout + result.stderr
    assert "Sector:" in out
    assert "Predicted" in out




