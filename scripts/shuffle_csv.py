import click
import pandas as pd


@click.command()
@click.option("-o", "--output", required=True, help="Destination file")
@click.option("-s", "--chunksize", default=10_000, help="Chunk size")
@click.argument('csv_files', nargs=-1)
def main(output, chunksize, csv_files):
    parsers = {f: pd.read_csv(f, chunksize=chunksize) for f in csv_files}
    header = True
    with open(output, "w") as outf:
        while parsers:
            to_remove = set()
            df_list = []
            for k, pr in parsers.items():
                try:
                    df = pr.read(chunksize)
                    df_list.append(df)
                except StopIteration:
                    to_remove.add(k)
            if not df_list:
                break
            df = pd.concat(df_list).sample(frac=1).reset_index(drop=True)
            df.to_csv(outf, header=header)
            header = False
            for k in to_remove:
                del parsers[k]


if __name__ == "__main__":
    main()
