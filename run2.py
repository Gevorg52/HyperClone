"""Launch the Apex Reality GUI 2.0 wizard (UI-1).

Serves the new `gui2/` wizard on 127.0.0.1:7861 (the legacy expert GUI stays on
`run.py` / :7860 until parity). See docs/GUI_REDESIGN.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui2.app import launch

if __name__ == "__main__":
    launch()
