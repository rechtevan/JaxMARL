import subprocess
import sys
import os
import pytest

def run_script(script_path, *args):
    result = subprocess.run([sys.executable, script_path, *args], capture_output=True, text=True)
    return result

@pytest.mark.skip(reason="JAX/Flax API incompatibility: jax 0.4.38 + flax 0.10.4 broken. See issue #20")
def test_script_with_arguments():
    script_path = os.path.join('baselines/MAPPO/mappo_rnn_smax.py')
    result = run_script(script_path, 'TOTAL_TIMESTEPS=1e4', 'WANDB_MODE=disabled')
    assert result.returncode == 0