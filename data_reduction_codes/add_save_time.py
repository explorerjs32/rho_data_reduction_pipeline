"""
Stamp each FITS file with the time it was saved to disk (UTC) as the SAVETIME header keyword.

Run this on the acquisition computer, on the original files, BEFORE copying them anywhere:
copying usually resets the file creation time, which would make SAVETIME wrong.

Usage:
    python add_save_time.py -d /path/to/night1 /path/to/night2        # searches sub-directories too
    python add_save_time.py -d /path/to/night1 --dry-run              # preview only
    python add_save_time.py -d /path/to/night1 --overwrite            # replace existing SAVETIME
    python add_save_time.py -d /path/to/night1 --use-mtime            # use last-modified time (no creation time)
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from astropy.io import fits


def creation_time(path, use_mtime=False):
    """Return the file creation time as a POSIX timestamp, or None if the platform can't provide it.

    If use_mtime is True, return the last-modified time instead. That is a good stand-in for the save
    time as long as the file has not been edited since it was written (editing headers updates it).
    """
    st = os.stat(path)
    if use_mtime:
        return st.st_mtime
    if hasattr(st, "st_birthtime"):          # macOS, Windows (Python 3.12+), some BSD
        return st.st_birthtime
    if sys.platform.startswith(("win", "cygwin")):   # older Python on Windows: st_ctime is creation time
        return st.st_ctime
    # Linux: os.stat has no birth time (st_ctime is the last metadata change, NOT creation).
    # GNU stat can read it through statx if the kernel and filesystem support it; it prints 0 or -
    # when the birth time is unknown.
    try:
        out = subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True, check=True).stdout.strip()
        birth = int(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None
    return birth if birth > 0 else None


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
    parser.add_argument("--use-mtime", action="store_true",
                        help="Use the file's last-modified time instead of its creation time "
                             "(fallback for systems that don't record creation time).")
    args = parser.parse_args()

    n_written = n_skipped = n_failed = 0

    for path in find_fits(args.dir):
        timestamp = creation_time(path, use_mtime=args.use_mtime)
        if timestamp is None:
            print(f"FAILED  {path}: no file creation time available on this system (try --use-mtime)")
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
