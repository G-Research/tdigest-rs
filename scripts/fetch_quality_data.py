"""Reproduce the pinned measured-data subsets; never accept changed upstream bytes."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "tests/quality/data/manifest.json"
    args.output.mkdir(parents=True, exist_ok=False)
    for source in json.loads(manifest.read_text()):
        with urllib.request.urlopen(source["download_url"], timeout=90) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != source["archive_sha256"]:
            raise RuntimeError("Upstream archive differs from the pinned source: " + source["id"])
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            raw = bundle.read(source["source_member"])
        if hashlib.sha256(raw).hexdigest() != source["source_sha256"]:
            raise RuntimeError("Upstream CSV differs from pinned source")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(source["columns"])
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        if len(rows) != source["rows"]:
            raise RuntimeError("Upstream row count differs")
        writer.writerows([row[c] for c in source["columns"]] for row in rows)
        payload = bytearray(gzip.compress(stream.getvalue().encode(), mtime=0))
        # Set the gzip OS byte to 255 so the fixture hash is the same on every platform.
        payload[9] = 255
        if hashlib.sha256(payload).hexdigest() != source["derived_sha256"]:
            raise RuntimeError("Derived bytes differ; review Python/zlib compatibility before updating fixtures")
        (args.output / source["derived_file"]).write_bytes(payload)


if __name__ == "__main__":
    main()
