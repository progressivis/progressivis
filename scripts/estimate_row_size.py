import sys
from progressivis.core.utils import estimate_row_size


def fun(file_: str) -> None:
    n_rows, row_size = estimate_row_size(file_)
    print(f"{file_=}: {n_rows=}, {row_size=}")


if __name__ == "__main__":
    for file_ in sys.argv[1:]:
        fun(file_)
