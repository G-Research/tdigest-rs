"""Local Maven repository for testing authenticated dependency resolution.

Serves staged candidate artifacts and caches public Maven Central dependencies.
Credentials are intentionally public fixture values, never production secrets.
Clients use only this repository; it is stopped before offline rechecks.
"""

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    repository = args.root.resolve()
    lock = threading.Lock()
    events = args.receipt.with_suffix(".jsonl")

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def log_message(self, *_):
            pass

        def serve(self, include_body):
            token = base64.b64encode(b"fixture:fixture-password").decode()
            if self.headers.get("Authorization") != "Basic " + token:
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="tdigest-fixture"')
                self.end_headers()
                return
            relative = self.path.removeprefix("/maven/")
            if (
                not self.path.startswith("/maven/")
                or ".." in relative
                or "?" in relative
                or Path(relative).is_absolute()
            ):
                self.send_error(400)
                return
            origin = "staged"
            try:
                path = (repository / relative).resolve()
                if path == repository or not path.is_relative_to(repository):
                    self.send_error(400)
                    return
                if not path.is_file():
                    origin = "https://repo.maven.apache.org/maven2/" + relative
                    data = urllib.request.urlopen(origin, timeout=45).read()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".partial-" + str(threading.get_ident()))
                    temporary.write_bytes(data)
                    temporary.replace(path)
                data = path.read_bytes()
            except urllib.error.HTTPError as error:
                self.send_error(error.code)
                return
            except (OSError, urllib.error.URLError) as error:
                self.send_error(502, type(error).__name__)
                return
            with lock:
                with events.open("a") as output:
                    output.write(
                        json.dumps(
                            {
                                "path": relative,
                                "origin": origin,
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "bytes": len(data),
                            }
                        )
                        + "\n"
                    )
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if include_body:
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    args.receipt.write_text(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{server.server_port}/maven",
                "credentials": "public fixture values",
                "production_publication": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.receipt, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
