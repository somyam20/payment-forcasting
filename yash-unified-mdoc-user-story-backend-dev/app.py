#!/usr/bin/env python3
"""
Entry point for Meeting Document Generator
Supports UI (Streamlit), CLI, and API (FastAPI) modes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # Check if running in CLI mode (has --video argument)
    if len(sys.argv) > 1 and "--video" in sys.argv:
        # Run CLI mode from main.py
        from main import main_cli
        main_cli()
    # Check for UI mode
    elif len(sys.argv) > 1 and "ui" in sys.argv:
        # Run UI mode (Streamlit)
        from src.frontend.streamlit_app import main
        main()
    else:
        # Default: Run FastAPI server (API mode)
        import uvicorn
        from api import app
        # Check if port is specified (e.g., python app.py 8080)
        port = 8000
        if len(sys.argv) > 1:
            try:
                port = int(sys.argv[1])
            except ValueError:
                pass  # Not a port number, use default
        uvicorn.run(app, host="0.0.0.0", port=port)

