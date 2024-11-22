"""
Local execution script.

Utility script for running the service locally for development and testing.
"""

import argparse
import sys
from pathlib import Path

import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import get_config


def main():
    """Run FastAPI service locally."""
    config = get_config()
    
    parser = argparse.ArgumentParser(description="Run model health monitor service locally")
    parser.add_argument("--host", type=str, default=config.get("api.host", "127.0.0.1"), help="Host address")
    parser.add_argument("--port", type=int, default=config.get("api.port", 8000), help="Port number")
    parser.add_argument("--reload", action="store_true", default=config.get("api.reload", False), help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
