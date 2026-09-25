import argparse
import os
import sys
import threading
import webbrowser

import uvicorn
from dotenv import load_dotenv

from web.config import ServerSettings
from web.observability import configure_logging
from web.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic research assistant (web UI).")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log query analysis, tool calls and timings")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    settings = ServerSettings.from_env()
    configure_logging(args.verbose, settings.log_format)
    if not settings.api_keys and args.host not in ("127.0.0.1", "localhost"):
        print("WARNING: listening on a public interface without APP_API_KEYS; anyone can use your Groq quota.")

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"Research assistant running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
