"""Make the repo root importable so `import harness...` resolves under pytest (tests/test_task_env.py)."""
import sys
from pathlib import Path

root = str(Path(__file__).parent)
if root not in sys.path:
    sys.path.insert(0, root)
