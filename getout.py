#!/usr/bin/env python3
"""

Reads the binary .dat recorded by the PCI-1710HG ADC (12-channel recorder at ORT)
and writes a text .out file in the same format as the usual format used by IPS people.

Key behaviour
-------------
• Automatically extracts timestamp (date & time) from the 20-byte file header.
• If no output filename is given, uses <input>.out.
• By default skips the first decoded row (to match original .out).
• Each block = 3080 B → 8-byte prefix + 3072 B ADC data (12 × 128 samples).

Usage
-----
python3 getout.py input.dat [output.out]
  --no-skip-first   keep first decoded row instead of skipping it
"""

import argparse, datetime, os, sys, numpy as np

# constants for format
HEADER_LEN = 20
BLOCK_RAW = 3080
BLOCK_PREFIX = 8
BLOCK_DATA = BLOCK_RAW - BLOCK_PREFIX
N_CHANNELS = 12
SAMPLES_PER_CHANNEL = 128
SAMPLE_PERIOD_MS = 20.0
YEAR_BASE = 1644  # offset base used in stored year field


def parse_args():
    p = argparse.ArgumentParser(description="Read the .dat file from from PCI-1710HG 12-ch ADC into .out text file")
    p.add_argument("infile", help="input .dat file")
    p.add_argument("outfile", nargs="?", help="optional output filename (.out)")
    p.add_argument("--no-skip-first", dest="skip_first", action="store_false",
                   help="do not drop first decoded row (default: skip it)")
    return p.parse_args()


def try_extract_datetime(bts):
    """Find plausible timestamp fields inside 20-byte header."""
    d = list(bts)
    L = len(d)
    for i in range(L - 8):
        try:
            day, mon0 = d[i], d[i + 1]
            yr = d[i + 2] | (d[i + 3] << 8)
            hr, mi, se = d[i + 4], d[i + 5], d[i + 6]
            ms = d[i + 7] | (d[i + 8] << 8) if i + 8 < L else d[i + 7]
            if 1 <= day <= 31 and 0 <= mon0 <= 11 and 0 <= hr <= 23 and 0 <= mi <= 59 and 0 <= se <= 59:
                y = YEAR_BASE + yr
                dt = datetime.datetime(y, mon0 + 1, day, hr, mi, se, ms * 1000)
                if 1970 <= y <= 2100:
                    return dt
        except Exception:
            pass
        
        # Try 1-byte year format (Base 1900)
        # Structure: Day, Mon, Yr(1byte), Hr, Min, Sec, MS(2bytes)
        try:
            day, mon0 = d[i], d[i + 1]
            yr_byte = d[i + 2]
            hr, mi, se = d[i + 3], d[i + 4], d[i + 5]
            ms = d[i + 6] | (d[i + 7] << 8) if i + 7 < L else d[i + 6]
            
            if 1 <= day <= 31 and 0 <= mon0 <= 11 and 0 <= hr <= 23 and 0 <= mi <= 59 and 0 <= se <= 59:
                y = 1900 + yr_byte
                dt = datetime.datetime(y, mon0 + 1, day, hr, mi, se, ms * 1000)
                if 1970 <= y <= 2100:
                    return dt
        except Exception:
            continue
    return None


def decode_dat(path):
    """Return (header_bytes, array[n_samples,12])"""
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        data = f.read()
    n_blocks = len(data) // BLOCK_RAW
    rows = []
    for b in range(n_blocks):
        blk = data[b * BLOCK_RAW:(b + 1) * BLOCK_RAW]
        clean = blk[BLOCK_PREFIX:BLOCK_PREFIX + BLOCK_DATA]
        w = np.frombuffer(clean, dtype="<u2").astype(np.int64)
        vals = (((w >> 8) & 0x0F) << 8) | (w & 0xFF)
        ch_major = vals.reshape(N_CHANNELS, SAMPLES_PER_CHANNEL)
        for s in range(SAMPLES_PER_CHANNEL):
            rows.append(ch_major[:, s].tolist())
    return header, np.array(rows, np.int64)


def format_header(dt):
    line1 = f"{dt.day} {dt.month} {dt.year}  {dt.hour} {dt.minute} {dt.second} {int(dt.microsecond/1000)}\n"
    line2 = f"No. of channels = {N_CHANNELS}      Samples per channel = {SAMPLES_PER_CHANNEL} (256 bytes)\n"
    line3 = f"ADC range : (-5.000000, 5.000000) Volts     Sampling time = {int(SAMPLE_PERIOD_MS)} ms.\n\n"
    return line1 + line2 + line3


def write_out(outpath, arr, start_dt, skip_first=True):
    start_index = 1 if skip_first else 0
    with open(outpath, "w") as f:
        f.write(format_header(start_dt))
        for i in range(start_index, len(arr)):
            t = start_dt + datetime.timedelta(milliseconds=SAMPLE_PERIOD_MS * (i - start_index + 1))
            hh, mm, ss, ms = t.hour, t.minute, t.second, int(t.microsecond / 1000)
            vals = " ".join(str(int(v)) for v in arr[i])
            f.write(f"{hh} {mm} {ss} {ms}  {vals}\n")


def main():
    a = parse_args()
    infile = a.infile
    outfile = a.outfile or os.path.splitext(infile)[0] + ".out"
    hdr, arr = decode_dat(infile)
    dt = try_extract_datetime(hdr) or datetime.datetime(2025, 1, 3, 1, 55, 24, 160000)
    print("Start time:", dt.isoformat(), "| samples:", arr.shape)
    write_out(outfile, arr, dt, a.skip_first)
    print("Wrote", outfile)


if __name__ == "__main__":
    main()

