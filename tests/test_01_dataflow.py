from pprint import pprint

from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset

from . import ProgressiveTest


class TestDataflow(ProgressiveTest):
    def test_dataflow(self):
        scheduler = self.scheduler()
        saved_inputs = None
        saved_outputs = None
        with scheduler.dataflow() as dataflow:
            csv = CSVLoader(get_dataset('smallfile'), name='csv',
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
            saved_inputs = dataflow.inputs
            saved_outputs = dataflow.outputs
            # dataflow.__exit__() is called here
        print('Old modules:', end=' ')
        pprint(scheduler._modules)
        scheduler._update_modules() # force modules in the main loop
        print('New modules:', end=' ')
        pprint(scheduler.modules())

        with scheduler.dataflow() as dataflow:
            # nothing should change when nothing is modified in dataflow
            self.assertEqual(len(dataflow), 3)
            deps = dataflow.order_modules()
            self.assertEqual(deps, ['csv', m.name, prt.name])
            self.assertEqual(dataflow.inputs, saved_inputs)
            self.assertEqual(dataflow.outputs, saved_outputs)
        scheduler._update_modules() # force modules in the main loop

        with scheduler.dataflow() as dataflow:
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
