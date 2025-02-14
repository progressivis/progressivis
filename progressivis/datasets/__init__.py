from __future__ import annotations

import os
import os.path

from functools import partial
from progressivis import ProgressiveError
from .random import (
    generate_random_csv,
    generate_random_parquet,
    generate_random_multivariate_normal_csv,
    generate_multiscale_random_csv,
)
from .wget import wget_file
import bz2
import zlib
import lzma
from typing import Any, Dict, cast, Type

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
Z_CHUNK_SIZE = 16 * 1024 * 32


def _check_kwds(kwds: Dict[str, Any], **defaults: Any) -> Dict[str, Any]:
    if ("rows" in kwds) or ("cols" in kwds) and ("overwrite" not in kwds):
        raise ValueError(
            "'owerwrite' param. is mandatory when " "'rows' or 'cols' is specified"
        )
    res = kwds.copy()
    for k, v in defaults.items():
        if k in res:
            continue
        res[k] = v
    return res


def get_dataset(name: str, **kwds: Any) -> str:
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    if name == "bigfile":
        kw = _check_kwds(kwds, rows=1_000_000, cols=30)
        return generate_random_csv(f"{DATA_DIR}/bigfile.csv", **kw)
    if name == "bigfile_parquet":
        n_cols = 30
        kw = _check_kwds(kwds, rows=1_000_000, cols=n_cols)
        csv_file = generate_random_csv(f"{DATA_DIR}/bigfile.csv", **kw)
        return generate_random_parquet(
            f"{DATA_DIR}/bigfile.parquet", csv_file, n_cols=n_cols
        )
    if name == "bigfile_multiscale":
        kw = _check_kwds(kwds, rows=5_000_000)
        return generate_multiscale_random_csv(
            f"{DATA_DIR}/bigfile_multiscale.csv", **kw
        )
    if name == "bigfile_mvn":
        kw = _check_kwds(kwds, rows=900_000)
        return generate_random_multivariate_normal_csv(
            f"{DATA_DIR}/bigfile_mvn.csv", **kw
        )
    if name == "smallfile":
        kw = _check_kwds(kwds, rows=30_000, cols=10)
        return generate_random_csv(f"{DATA_DIR}/smallfile.csv", **kw)
    if name == "smallfile_parquet":
        n_cols = 10
        kw = _check_kwds(kwds, rows=30_000, cols=n_cols)
        csv_file = generate_random_csv(f"{DATA_DIR}/smallfile.csv", **kw)
        return generate_random_parquet(
            f"{DATA_DIR}/smallfile.parquet", csv_file, n_cols=n_cols
        )
    if name == "smallfile_multiscale":
        kw = _check_kwds(kwds, rows=30_000)
        return generate_multiscale_random_csv(
            f"{DATA_DIR}/smallfile_multiscale.csv", **kw
        )

    if name == "warlogs":
        return wget_file(
            filename=f"{DATA_DIR}/warlogs.vec.bz2",
            url="http://www.cs.ubc.ca/labs/imager/video/2014/QSNE/warlogs.vec.bz2"
        )
    if name == "mnist_784":
        # This file [mnist_784.csv] is made available under the Public Domain
        # Dedication and License v1.0 whose full text can be found at: http://opendatacommons.org/licenses/pddl/1.0/
        return wget_file(
            filename=f"{DATA_DIR}/mnist_784.csv",
            url="https://datahub.io/machine-learning/mnist_784/r/mnist_784.csv"
        )
    if name.startswith("cluster:"):
        fname = name[len("cluster:") :] + ".txt"
        return wget_file(
            filename=f"{DATA_DIR}/{fname}",
            url=f"http://cs.joensuu.fi/sipu/datasets/{fname}",
        )
    raise ProgressiveError(f"Unknown dataset {name}")


class Compressor:
    def compress(self, data: bytes) -> bytes:
        return None  # type: ignore

    def flush(self) -> bytes:
        return None  # type: ignore


compressors: Dict[str, Dict[str, Any]] = dict(
    bz2=dict(ext=".bz2", factory=bz2.BZ2Compressor),
    zlib=dict(ext=".zlib", factory=zlib.compressobj),
    gzip=dict(ext=".gz", factory=partial(zlib.compressobj, wbits=zlib.MAX_WBITS | 16)),
    lzma=dict(ext=".xz", factory=lzma.LZMACompressor),
)


def get_dataset_compressed(
    name: str, compressor: Dict[str, Compressor], **kwds: Any
) -> str:
    source_file = get_dataset(name, **kwds)
    dest_file = source_file + cast(str, compressor["ext"])
    if os.path.isfile(dest_file):
        return dest_file
    factory = cast(Type[Compressor], compressor["factory"])
    comp = factory()
    with open(source_file, "rb") as rdesc:
        with open(dest_file, "wb") as wdesc:
            while True:
                data = rdesc.read(Z_CHUNK_SIZE)
                if not data:
                    break
                wdesc.write(comp.compress(data))
            wdesc.write(comp.flush())
    return dest_file


def get_dataset_bz2(name: str, **kwds: Any) -> str:
    return get_dataset_compressed(name, compressors["bz2"], **kwds)


def get_dataset_zlib(name: str, **kwds: Any) -> str:
    return get_dataset_compressed(name, compressors["zlib"], **kwds)


def get_dataset_gz(name: str, **kwds: Any) -> str:
    return get_dataset_compressed(name, compressors["gzip"], **kwds)


def get_dataset_lzma(name: str, **kwds: Any) -> str:
    return get_dataset_compressed(name, compressors["lzma"], **kwds)


__all__ = [
    "get_dataset",
    "get_dataset_bz2",
    "get_dataset_zlib",
    "get_dataset_gz",
    "get_dataset_lzma",
    "generate_random_csv",
]
