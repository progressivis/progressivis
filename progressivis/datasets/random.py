from __future__ import annotations

import numpy as np
import csv
import os
import os.path
import pandas as pd


from typing import Optional, Any, List, Sequence


# filename='data/bigfile.csv'
# rows = 1000000
# cols = 30


def generate_random_csv(
    filename: str, rows: int = 900_000, cols: int = 10, seed: int = 1234
) -> str:
    if os.path.exists(filename):
        return filename
    try:
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            np.random.seed(seed=seed)
            for _ in range(0, rows):
                row = list(np.random.normal(loc=0.5, scale=0.8, size=cols))
                writer.writerow(row)
    except (KeyboardInterrupt, SystemExit):
        os.remove(filename)
        raise
    return filename


def generate_random_parquet(filename: str, csv_file: str, n_cols: int) -> str:
    if os.path.exists(filename):
        return filename
    df = pd.read_csv(csv_file, names=[f"_{i}" for i in range(n_cols)])
    df.to_parquet(filename)
    return filename


def generate_random_multivariate_normal_csv(
    filename: str,
    rows: int,
    seed: int = 1234,
    header: Optional[str] = None,
    reset: Optional[bool] = False,
) -> str:
    """
    Adapted from: https://github.com/e-/PANENE/blob/master/examples/kernel_density/online.py
    Author: Jaemin Jo
    Date: February 2019
    License: https://github.com/e-/PANENE/blob/master/LICENSE
    """
    if isinstance(filename, str) and os.path.exists(filename) and not reset:
        return filename

    def mv(
        n: int, mean: List[float], cov: List[List[float]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.random.multivariate_normal(mean, cov, size=(n)).astype(np.float32)

    N = rows // 3
    X = np.concatenate(
        (
            mv(N, [0.1, 0.3], [[0.01, 0], [0, 0.09]]),
            mv(N, [0.7, 0.5], [[0.04, 0], [0, 0.01]]),
            mv(N, [-0.4, -0.3], [[0.09, 0.04], [0.04, 0.02]]),
        ),
        axis=0,
    )
    np.random.shuffle(X)
    kw = {} if header is None else dict(header=header, comments="")
    np.savetxt(filename, X, delimiter=",", **kw)
    return filename


def generate_multiscale_random_csv(
    filename: str,
    rows: int = 5_000_000,
    seed: int = 1234,
    choice: Sequence[str] = ("A", "B", "C", "D"),
    overwrite: bool = False,
) -> str:
    if os.path.exists(filename):
        return filename
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 3, rows),
            "B": np.random.normal(5, 2, rows),
            "C": np.random.normal(-5, 4, rows),
            "D": np.random.normal(5, 3, rows),
            "I": np.random.randint(0, 10_000, size=rows, dtype=int),
            "S": np.random.choice(choice, rows),
        }
    )
    try:
        df.to_csv(filename, index=False)
    except (KeyboardInterrupt, SystemExit):
        os.remove(filename)
        raise
    return filename
