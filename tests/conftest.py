import sys
from pathlib import Path

# Same pattern used by scripts/eval.py etc. -- makes `from backend...` imports resolve
# regardless of the directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
