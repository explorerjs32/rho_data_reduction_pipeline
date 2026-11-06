"""
Stamp each FITS file with the time it was saved to disk (UTC) as the SAVETIME header keyword.

Run this on the acquisition computer, on the original files, BEFORE copying them anywhere:
copying usually resets the file creation time, which would make SAVETIME wrong.

Usage:
    python add_save_time.py -d /path/to/night1 /path/to/night2        # searches sub-directories too
    python add_save_time.py -d /path/to/night1 --dry-run              # preview only
    python add_save_time.py -d /path/to/night1 --overwrite            # replace existing SAVETIME
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from astropy.io import fits


def creation_time(path):
    """Return the file creation time as a POSIX timestamp, or None if the platform can't provide it."""
    st = os.stat(path)
    if hasattr(st, "st_birthtime"):          # macOS, Windows (Python 3.12+), some BSD/Linux
        return st.st_birthtime
    if sys.platform.startswith("win"):       # older Python on Windows: st_ctime is creation time
        return st.st_ctime
    return None                              # Linux: st_ctime is NOT creation time


def find_fits(directories):
    for directory in directories:
        for root, _, files in os.walk(directory):
            for name in sorted(files):
                if name.lower().endswith((".fits", ".fit", ".fts")):
                    yield os.path.join(root, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dir", nargs="+", required=True,
                        help="One or more observation directories (searched recursively).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without modifying files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace SAVETIME if it already exists.")
    args = parser.parse_args()

    n_written = n_skipped = n_failed = 0

    for path in find_fits(args.dir):
        timestamp = creation_time(path)
        if timestamp is None:
            print(f"FAILED  {path}: this platform does not expose a file creation time")
            n_failed += 1
            continue

        # Timestamps are absolute (seconds since the epoch), so the conversion to UTC is correct
        # as long as the computer's clock and time zone were set correctly when the data were taken.
        save_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        if args.dry_run:
            print(f"WOULD SET {path}: SAVETIME={save_time}")
            continue

        with fits.open(path, mode="update") as hdul:
            header = hdul[0].header
            if "SAVETIME" in header and not args.overwrite:
                print(f"SKIPPED {path}: SAVETIME already set ({header['SAVETIME']})")
                n_skipped += 1
                continue
            header["SAVETIME"] = (save_time, "file creation date UTC")
        print(f"SET     {path}: SAVETIME={save_time}")
        n_written += 1

    print(f"\nDone: {n_written} written, {n_skipped} skipped, {n_failed} failed.")


if __name__ == "__main__":
    main()
