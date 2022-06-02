import click
from datetime import date
import os
import os.path as osp
import pyarrow.parquet as pq
from progressivis.datasets.wget import wget_file

KILOBYTE = 1 << 10
MEGABYTE = KILOBYTE ** 2
URL = "https://s3.amazonaws.com/nyc-tlc/trip+data"
FILE = "{transp}_tripdata_{year}-{month:0>2}.parquet"
TRANSPORTS = ["yellow", "green", "fhv", "fhvhv"]


def _validate_size(ctx, param, value):
    try:
        return int(value), value
    except ValueError:
        try:
            num = int(value[:-1])
            end = value[-1]
            times = dict(k=KILOBYTE, M=MEGABYTE)[end]
            return num * times, value
        except Exception:
            raise click.BadParameter("format must be <int>|<int>k|<int>M")


@click.command()
@click.option(
    "-y",
    "--year",
    default=2015,
    type=click.IntRange(2009, date.today().year),
    help="Year (default: 2015)",
)
@click.option(
    "-m",
    "--month",
    default=list(range(1, 13)),
    type=click.IntRange(1, 12),
    multiple=True,
    help="default: 1..12",
)
@click.option(
    "-s",
    "--row_group_size",
    default="500k",
    callback=_validate_size,
    help="[<int>|<int>k|<int>M] (default: 500k)",
)
@click.option(
    "-t",
    "--transport",
    type=click.Choice(TRANSPORTS),
    multiple=True,
    default=["yellow", "green"],
    help="default: yellow+green",
)
@click.option("-d", "--dest", default="nyc-taxi", help="Destination directory")
@click.option("-p", "--prefix", default="pq", help="Output file prefix (default: pq_)")
def main(year, month, row_group_size, transport, dest, prefix):
    here = osp.dirname(osp.abspath(__file__))
    repo_root = osp.dirname(here)
    data_dir = osp.join(repo_root, dest)
    if not osp.isdir(data_dir):
        raise ValueError(f"{data_dir} does not exist or is not a directory, abort")
    tmp_file = f"{data_dir}/tmp.parquet"
    if osp.exists(tmp_file):
        raise ValueError(f"{tmp_file} already exists, abort")
    today = date.today()
    size, raw = row_group_size
    current_year = today.year
    months = sorted(set(month))
    transports = [tr for tr in TRANSPORTS if tr in transport]
    if year == current_year:
        months = [m for m in months if m < today.month]
    for tr in transports:
        for mo in months:
            aws_file = FILE.format(year=year, month=mo, transp=tr)
            chunked_file = f"{data_dir}/{prefix}_{raw}_{aws_file}"
            url = f"{URL}/{aws_file}"
            try:
                if osp.exists(chunked_file):
                    raise ValueError(f"{chunked_file} already exists")
                wget_file(filename=tmp_file, url=url)
                table = pq.read_table(tmp_file)
                pq.write_table(table, chunked_file, row_group_size=size)

            except Exception as ee:
                print(ee, url)
            finally:
                if osp.exists(tmp_file):
                    os.remove(tmp_file)


if __name__ == "__main__":
    main()
