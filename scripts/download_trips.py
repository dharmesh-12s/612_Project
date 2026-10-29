"""Download Capital Bikeshare monthly trip files (2021-2025) into data/raw/trips/.

Usage:
    python scripts/download_trips.py                  # all of 2021-2025
    python scripts/download_trips.py --start 202401 --end 202402   # small test

Source: https://capitalbikeshare.com/system-data (public S3 bucket).
"""
import argparse
import re
import zipfile
from pathlib import Path

import requests

BUCKET = "https://s3.amazonaws.com/capitalbikeshare-data/"
OUT_DIR = Path("data/raw/trips")


def list_bucket():
    """Return all file names in the bucket (used to find the exact monthly names)."""
    keys, token = [], None
    while True:
        params = {"list-type": "2"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def months(start, end):
    y, m = divmod(start, 100)
    while y * 100 + m <= end:
        yield f"{y}{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=202101)
    ap.add_argument("--end", type=int, default=202512)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = [k for k in list_bucket() if k.endswith(".zip")]

    for ym in months(args.start, args.end):
        matches = [k for k in keys if k.startswith(ym)]
        if not matches:
            print(f"[missing] {ym}: no file in bucket")
            continue
        for key in matches:
            if any(OUT_DIR.glob(f"{ym}*.csv")):
                print(f"[skip] {ym}: already extracted")
                break
            print(f"[get] {key}")
            zpath = OUT_DIR / key
            with requests.get(BUCKET + key, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            with zipfile.ZipFile(zpath) as z:
                for name in z.namelist():
                    # skip macOS junk folders inside some zips
                    if name.endswith(".csv") and not name.startswith("__MACOSX"):
                        z.extract(name, OUT_DIR)
            zpath.unlink()


if __name__ == "__main__":
    main()
