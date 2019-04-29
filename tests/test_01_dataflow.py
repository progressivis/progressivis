from pprint import pprint

from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset

from . import ProgressiveTest


class TestDataflow(ProgressiveTest):
    def test_dataflow(self):
        dataflow = self.dataflow()
        csv = CSVLoader(get_dataset('bigfile'), name='csv',
                        index_col=False, header=None,
                        dataflow=dataflow)
        self.assertIs(dataflow['csv'], csv)

        m = Min(dataflow=dataflow)
        self.assertIs(dataflow[m.name], m)

        prt = Print(proc=self.terse,
                    dataflow=dataflow)
        self.assertIs(dataflow[prt.name], prt)

        m.input.table = csv.output.table
        prt.input.df = m.output.table

        self.assertEqual(len(dataflow), 3)
        deps = dataflow.order_modules()
        self.assertEqual(deps, ['csv', m.name, prt.name])


if __name__ == '__main__':
    ProgressiveTest.main()
