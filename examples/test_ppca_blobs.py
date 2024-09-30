from progressivis import Scheduler, Every
from progressivis.stats.ppca import PPCA
from progressivis.core import aio
from progressivis.stats.blobs_table import BlobsTable
import click

try:
    s = scheduler
except NameError:
    s = Scheduler()


def _print(x):
    pass


@click.command()
@click.option('--n_samples', default=10000000, help='Number of samples')
@click.option('--n_components', default=2, help='Principal components number')
@click.option('--rtol', default=0.3, help='relative tolerance')
@click.option('--trace', default='verbose', help='trace')
@click.option('--csv_log_file', default=None, help='Save trace as a csv file')
def main(n_samples, n_components, rtol, trace, csv_log_file):
    centers = [(0.1, 0.3, 0.5), (0.7, 0.5, 0.3), (-0.4, -0.3, -0.1)]
    data = BlobsTable(columns=['_0', '_1', '_2'], centers=centers,
                      cluster_std=0.2, rows=n_samples, scheduler=s)
    ppca = PPCA(scheduler=s)
    ppca.input.table = data.output.table
    ppca.params.n_components = n_components
    ppca.create_dependent_modules(rtol=rtol, trace=trace)
    prn = Every(scheduler=s, proc=_print)
    prn.input.df = ppca.reduced.output.table
    aio.run(s.start())
    if csv_log_file:
        ppca.reduced._trace_df.to_csv(csv_log_file, index=False)


if __name__ == '__main__':
    main()
