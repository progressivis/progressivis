from pprint import pprint

from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset

from . import ProgressiveTest


class TestDataflow(ProgressiveTest):
    def test_dataflow(self):
        scheduler = self.scheduler()
        with scheduler.dataflow() as dataflow:
            csv = CSVLoader(get_dataset('bigfile'), name='csv',
                            index_col=False, header=None,
                            dataflow=dataflow)
            self.assertIs(dataflow['csv'], csv)
            self.assertEqual(dataflow.validate_module(csv), [])

            m = Min(name="min", dataflow=dataflow)
            self.assertIs(dataflow[m.name], m)
            self.assertEqual(dataflow.validate_module(m),
                             ['Input slot "table" missing in module "min"'])

            prt = Print(proc=self.terse,
                        name="print",
                        dataflow=dataflow)
            self.assertIs(dataflow[prt.name], prt)
            self.assertEqual(dataflow.validate_module(prt),
                             ['Input slot "df" missing in module "print"'])

            m.input.table = csv.output.table
            prt.input.df = m.output.table

            self.assertEqual(len(dataflow), 3)
            errors = dataflow.validate()
            self.assertEqual(errors, [])
            deps = dataflow.order_modules()
            self.assertEqual(deps, ['csv', m.name, prt.name])

            dataflow.remove_module(prt)
            self.assertEqual(len(dataflow), 2)
            deps = dataflow.order_modules()
            self.assertEqual(deps, ['csv', m.name])
            pprint(dataflow.inputs)
            pprint(dataflow.outputs)
            # dataflow.__exit__() is called here
        print('Old modules:')
        pprint(scheduler._new_modules)
        scheduler._update_modules() # force modules in the main loop
        print('New modules:')
        pprint(scheduler.modules())
        



if __name__ == '__main__':
    ProgressiveTest.main()
