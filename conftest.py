"""Make the repo root importable so `import training...` resolves under pytest."""
import sys
from pathlib import Path

root = str(Path(__file__).parent)
if root not in sys.path:
    sys.path.insert(0, root)
