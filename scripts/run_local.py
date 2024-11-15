"""
Local execution script.

Utility script for running the service locally for development and testing.
"""

import argparse
from pathlib import Path

import uvicorn


def main():
    """Run FastAPI service locally."""
    parser = argparse.ArgumentParser(description="Run model health monitor service locally")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
