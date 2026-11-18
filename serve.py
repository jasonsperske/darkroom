#!/usr/bin/env python3
"""Local development server for web/ that disables caching.

`python3 -m http.server` lets browsers cache module and worker scripts heuristically, so after
editing a file you can end up running stale code inside a Web Worker. This sends
`Cache-Control: no-store` on everything. Usage: ./serve.py [port]   (default 8000)
"""
import functools
import http.server
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(NoCacheHandler, directory=__file__.rsplit("/", 1)[0] + "/web")
    print(f"Serving web/ at http://localhost:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
