import os


# Use command line with subprocess to run pytest
import subprocess
def test_psiformer():
    result = subprocess.run(['pytest', 'src/psiformer_torch/tests/mc_test.py'], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0