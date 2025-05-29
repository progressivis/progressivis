from progressivis import Scheduler, Every, CSVLoader
from progressivis.datasets import get_dataset
from progressivis.stats.ppca import PPCA
from progressivis.core import aio
import click
try:
    s = scheduler
except NameError:
    s = Scheduler()


def _print(x):
    pass


@click.command()
@click.option('--dataset', default='mnist_784', help='Input dataset')
@click.option('--n_components', default=16, help='Principal components number')
@click.option('--rtol', default=0.3, help='relative tolerance')
@click.option('--trace', default='verbose', help='trace')
@click.option('--csv_log_file', default=None, help='Save trace as a csv file')
def main(dataset, n_components, rtol, trace, csv_log_file):
    if not dataset.endswith('.csv'):
        dataset = get_dataset(dataset)
    data = CSVLoader(dataset,
                     usecols=lambda x: x != 'class', scheduler=s)
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
