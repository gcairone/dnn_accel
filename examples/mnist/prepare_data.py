"""Downloads real MNIST (the ossci-datasets S3 mirror, the same source
torchvision uses internally) and writes it out as raw uint8 binary files
with the IDX header stripped -- so train_cpp.cpp doesn't need a gzip/zlib
dependency just for a demo, and all three example implementations
(train_cpp.cpp, train_fastmlp.py, train_torch.py) read byte-identical
images/labels.

Run once (writes into examples/mnist/data/, not checked in -- ~55MB,
regenerated on demand like any other build artifact):
    python3 examples/mnist/prepare_data.py
"""
import gzip
import os
import struct
import urllib.request

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_gz(name):
    path = os.path.join(OUT_DIR, FILES[name])
    if not os.path.exists(path):
        print(f"  downloading {FILES[name]} ...")
        urllib.request.urlretrieve(BASE + FILES[name], path)
    return path


def strip_images(gz_path, out_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic in {gz_path}"
        data = f.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return n, rows, cols


def strip_labels(gz_path, out_path):
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic in {gz_path}"
        data = f.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_tr, rows, cols = strip_images(fetch_gz("train_images"), os.path.join(OUT_DIR, "train_images.bin"))
    n_tr_l = strip_labels(fetch_gz("train_labels"), os.path.join(OUT_DIR, "train_labels.bin"))
    n_te, _, _ = strip_images(fetch_gz("test_images"), os.path.join(OUT_DIR, "test_images.bin"))
    n_te_l = strip_labels(fetch_gz("test_labels"), os.path.join(OUT_DIR, "test_labels.bin"))
    assert n_tr == n_tr_l and n_te == n_te_l
    print(f"wrote {n_tr} train / {n_te} test images ({rows}x{cols}) as raw uint8 to {OUT_DIR}")
    print("train_images.bin/test_images.bin: N*784 uint8 row-major, no header")
    print("train_labels.bin/test_labels.bin: N uint8, no header")


if __name__ == "__main__":
    main()
