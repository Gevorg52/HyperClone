"""Apex Reality — SfM Pipeline MVP entry point."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.app import launch

if __name__ == "__main__":
    launch()
