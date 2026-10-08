"""Split an assembled (uncompressed) dataset .npz back into ``run_chunked`` sidecar chunks.

Lets a finished dataset be EXTENDED with the same seed: ``run_chunked`` seeds every chunk from
``(seed, row_offset)`` alone, so a run with a larger ``data.n_simulations`` (same seed and
chunk_size) regenerates the first chunks bit-identically. Pre-populating
``<new_stem>.chunks/chunk_XXXXX.npz`` from the old file makes the resume logic skip them and
simulate only the new rows.

    python scripts/split_npz_to_chunks.py OLD.npz NEW_STEM.chunks --chunk-size 1000

Reads each member through a memmap (the archive is ZIP_STORED, as ``np.savez`` writes it), so
peak memory is one chunk. Chunks are written atomically in the same format as
``io._save_chunk_atomic``; existing chunks are skipped, so the script is resumable.
"""

import argparse
import os
import zipfile

import numpy as np
from tqdm import tqdm


def member_memmaps(path):
    """{key: read-only memmap} for every .npy member of an uncompressed .npz."""
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{info.filename} is compressed; cannot memmap")
            fh.seek(info.header_offset)
            local = fh.read(30)
            name_len = int.from_bytes(local[26:28], "little")
            extra_len = int.from_bytes(local[28:30], "little")
            data_start = info.header_offset + 30 + name_len + extra_len
            fh.seek(data_start)
            version = np.lib.format.read_magic(fh)
            read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read(fh)
            if dtype.hasobject:
                raise ValueError(f"{info.filename} holds Python objects; cannot memmap")
            key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            out[key] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape,
                                 order="F" if fortran else "C")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("chunk_dir")
    ap.add_argument("--chunk-size", type=int, default=1000)
    args = ap.parse_args()

    arrays = member_memmaps(args.src)
    n = {len(a) for a in arrays.values()}
    if len(n) != 1:
        raise ValueError(f"inconsistent leading axes: { {k: a.shape for k, a in arrays.items()} }")
    n = n.pop()
    if n % args.chunk_size:
        raise ValueError(f"{n} rows is not a multiple of chunk_size={args.chunk_size}")
    os.makedirs(args.chunk_dir, exist_ok=True)
    for idx in tqdm(range(n // args.chunk_size), desc="splitting", unit="chunk"):
        path = os.path.join(args.chunk_dir, f"chunk_{idx:05d}.npz")
        if os.path.exists(path):
            continue
        sl = slice(idx * args.chunk_size, (idx + 1) * args.chunk_size)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **{k: np.ascontiguousarray(a[sl]) for k, a in arrays.items()})
        os.replace(tmp, path)


if __name__ == "__main__":
    main()
