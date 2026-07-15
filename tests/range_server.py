#!/usr/bin/env python3
# Minimal static file server WITH HTTP Range support, for the download tests. The stdlib
# http.server ignores `Range:` and always returns the whole file (200), so it can't exercise
# the segmented (parallel byte-range) downloader — a real CDN like Hugging Face answers each
# range with 206 + Content-Range, which is what this emulates. HEAD still reports Content-Length
# (used to resolve size), and non-range GETs fall back to the stdlib handler.
#
# Usage: range_server.py <port> <directory>
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d+)-(\d*)$", rng) if rng else None
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().do_GET()                      # no/!parseable range -> full file
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_error(416)
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    port = int(sys.argv[1])
    directory = sys.argv[2]
    handler = partial(RangeHandler, directory=directory)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
