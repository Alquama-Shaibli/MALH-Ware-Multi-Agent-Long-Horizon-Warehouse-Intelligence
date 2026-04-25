"""
app.py — HuggingFace Space Entry Point
=======================================
This file lives at the project ROOT so HF Spaces can auto-detect it.
It imports the FastAPI app from server.app and runs uvicorn on port 7860
(the port required by all HF Docker Spaces).

If you are running locally, the server/app.py __main__ block still uses
port 8000 — only HF Spaces uses this file.
"""

import sys
import os

# Ensure the project root is on sys.path so warehouse_env is importable
# regardless of which working directory HF Spaces uses.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from server.app import app  # noqa: E402  (after sys.path fix)


def main() -> None:
    """Entry point for HuggingFace Spaces (port 7860)."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
