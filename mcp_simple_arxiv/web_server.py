"""
Web server entry point for the MCP server using HTTP transport.
"""

import argparse
import logging
from mcp_simple_arxiv.server import app


def parse_args():
    """Parse command-line arguments for host/port overrides."""
    parser = argparse.ArgumentParser(description="Run the mcp-simple-arxiv web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    return parser.parse_args()


def main():
    """Run the MCP server as a web server."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    app = create_app()
    app.run(transport="streamable-http", host=args.host, port=args.port,stateless_http=True)


if __name__ == "__main__":
    main()
