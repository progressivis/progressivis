from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Min, Max, RandomTable
from progressivis.datasets import get_dataset
from progressivis.core import aio, SlotDescriptor, Module, Sink
from progressivis.core.dataflow import Dataflow

from . import ProgressiveTest


class TestModule(Module):
    inputs = [SlotDescriptor("a"), SlotDescriptor("b", required=False)]
    outputs = [SlotDescriptor("c"), SlotDescriptor("d", required=False)]

    def __init__(self, **kwds):
        super(TestModule, self).__init__(**kwds)

    def run_step(self, run_number, step_size, howlong):  # pragma no cover
        return self._return_run_step(self.state_blocked, 0)


class TestDataflow(ProgressiveTest):
    def test_dataflow_0(self):
        scheduler = self.scheduler()
        saved_inputs = None
        saved_outputs = None
        with scheduler as dataflow:
            csv = CSVLoader(
                get_dataset("smallfile"),
                name="csv",
                index_col=False,
                header=None,
                scheduler=scheduler,
            )
            self.assertIs(scheduler["csv"], csv)
            self.assertEqual(
                dataflow.validate_module(csv),
                ['Output slot "result" missing in module "csv"'],
            )

            m = Min(name="min", scheduler=scheduler)
            self.assertIs(dataflow[m.name], m)
            self.assertEqual(
                dataflow.validate_module(m),
                [
                    'Input slot "table" missing in module "min"',
                    'Output slot "result" missing in module "min"',
                ],
            )

            prt = Print(proc=self.terse, name="print", scheduler=scheduler)
            self.assertIs(dataflow[prt.name], prt)
            self.assertEqual(
                dataflow.validate_module(prt),
                ['Input slot "df" missing in module "print"'],
            )

            m.input.table = csv.output.result
            prt.input.df = m.output.result

            self.assertEqual(len(dataflow), 3)
            self.assertEqual(dataflow.dir(), ["csv", "min", "print"])
            errors = dataflow.validate()
            self.assertEqual(errors, [])
            deps = dataflow.order_modules()
            self.assertEqual(deps, ["csv", m.name, prt.name])
            saved_inputs = dataflow.inputs
            saved_outputs = dataflow.outputs
            # dataflow.__exit__() is called here
        # print('Old modules:', end=' ')
        # pprint(scheduler._modules)
        scheduler._update_modules()  # force modules in the main loop
        # print('New modules:', end=' ')
        # pprint(scheduler.modules())

        with scheduler as dataflow:
            # nothing should change when nothing is modified in dataflow
            self.assertEqual(len(dataflow), 3)
            deps = dataflow.order_modules()
            self.assertEqual(deps, ["csv", m.name, prt.name])
            self.assertEqual(dataflow.inputs, saved_inputs)
            self.assertEqual(dataflow.outputs, saved_outputs)
        scheduler._update_modules()  # force modules in the main loop

        with scheduler as dataflow:
            sink = Sink(name="sink", scheduler=scheduler)
            sink.input.inp = m.output.result
            dataflow.delete_modules(prt)
            self.assertEqual(len(dataflow), 3)
            deps = dataflow.order_modules()
            self.assertEqual(deps, ["csv", m.name, "sink"])
            # pprint(dataflow.inputs)
            # pprint(dataflow.outputs)
        # print('Old modules:')
        # pprint(scheduler._new_modules)
        scheduler._update_modules()  # force modules in the main loop
        # print('New modules:')
        # pprint(scheduler.modules())
        with scheduler as dataflow:
            self.assertEqual(len(dataflow), 3)
            deps = dataflow.order_modules()
            self.assertEqual(deps, ["csv", m.name, "sink"])
            prt = Print(proc=self.terse, name="print", scheduler=scheduler)
            self.assertIs(dataflow[prt.name], prt)
            self.assertEqual(
                dataflow.validate_module(prt),
                ['Input slot "df" missing in module "print"'],
            )

            prt.input.df = m.output.result
        scheduler._update_modules()  # force modules in the main loop

    def test_dataflow_1_dynamic(self):
        scheduler = self.scheduler(clean=True)

        csv = CSVLoader(
            get_dataset("bigfile"),
            name="csv",
            index_col=False,
            header=None,
            scheduler=scheduler,
        )
        m = Min(name="min", scheduler=scheduler)
        prt = Print(proc=self.terse, name="print_min", scheduler=scheduler)
        m.input.table = csv.output.result
        prt.input.df = m.output.result
        started = False

        def proc(x):
            nonlocal started
            print("proc max called")
            started = True

        async def _add_max(csv, scheduler, proc):
            await aio.sleep(2)
            with scheduler:
                print("adding new modules")
                m = Max(name="max", scheduler=scheduler)
                prt = Print(name="print_max", proc=proc, scheduler=scheduler)
                m.input.table = csv.output.result
                prt.input.df = m.output.result

        t = _add_max(csv, scheduler, proc=proc)
        aio.run_gather(scheduler.start(), t)
        self.assertTrue(started)

    def test_dataflow_2_add_remove(self):
        scheduler = self.scheduler(clean=True)

        csv = CSVLoader(
            get_dataset("bigfile"),
            name="csv",
            index_col=False,
            header=None,
            scheduler=scheduler,
        )
        m = Min(name="min", scheduler=scheduler)
        prt = Print(proc=self.terse, name="print_min", scheduler=scheduler)
        m.input.table = csv.output.result
        prt.input.df = m.output.result
        started = False

        def proc(x):
            nonlocal started
            print("proc max called")
            started = True

        async def _add_max_remove_min(csv, scheduler, proc):
            await aio.sleep(2)
            with scheduler as dataflow:
                print("adding new modules")
                m = Max(name="max", scheduler=scheduler)
                prt = Print(name="print_max", proc=proc, scheduler=scheduler)
                m.input.table = csv.output.result
                prt.input.df = m.output.result
                print("removing min module")
                dataflow.delete_modules("min", "print_min")

        t = _add_max_remove_min(csv, scheduler, proc=proc)
        aio.run_gather(scheduler.start(), t)
        self.assertTrue(started)

    def test_dataflow_3_dels(self):
        s = self.scheduler()
        table = RandomTable(name="table", columns=["a"], scheduler=s)
        m = Min(name="min", scheduler=s)
        m.input.table = table.output.result
        prt = Print(name="prt", scheduler=s)
        prt.input.df = m.output.result

        aio.run(s.step())
        with s as dataflow:
            self.assertTrue(isinstance(dataflow, Dataflow))
            deps = dataflow.collateral_damage("table")
            self.assertEquals(deps, set(["table", "min", "prt"]))

    def test_dataflow_4_dels2(self):
        s = self.scheduler()
        table = RandomTable(name="table", columns=["a"], scheduler=s)
        m = TestModule(name="min", scheduler=s)
        m.input.a = table.output.result
        prt = Print(name="prt", scheduler=s)
        prt.input.df = m.output.c
        # from nose.tools import set_trace; set_trace()
        s.commit()
        aio.run(s.step())
        with s as dataflow:
            self.assertTrue(isinstance(dataflow, Dataflow))
            deps = dataflow.collateral_damage("table")
            self.assertEquals(deps, set(["table", "min", "prt"]))

    def test_dataflow_5_dels_opt(self):
        s = self.scheduler()
        table = RandomTable(name="table", columns=["a"], scheduler=s)
        m = TestModule(name="min", scheduler=s)
        m.input.a = table.output.result
        prt = Print(name="prt", scheduler=s)
        prt.input.df = m.output.c
        prt2 = Print(name="prt2", scheduler=s)
        prt2.input.df = m.output.c
        s.commit()
        aio.run(s.step())
        with s as dataflow:
            self.assertTrue(isinstance(dataflow, Dataflow))
            deps = dataflow.collateral_damage("prt2")
            self.assertEquals(deps, set(["prt2"]))
            deps = dataflow.collateral_damage("prt")
            self.assertEquals(deps, set(["prt"]))
            deps = dataflow.collateral_damage("prt", "prt2")
            self.assertEquals(deps, set(["prt", "prt2", "min", "table"]))

    def test_dataflow_6_dynamic(self):
        s = self.scheduler()
        table = RandomTable(name="table", columns=["a"], scheduler=s)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = table.output.result
        prt = Print(name="prt", scheduler=s)
        prt.input.df = table.output.result
        prt2 = Print(name="prt2", scheduler=s)
        prt2.input.df = table.output.result
        s.commit()
        aio.run(s.step())
        # from nose.tools import set_trace; set_trace()
        with s as dataflow:
            self.assertTrue(isinstance(dataflow, Dataflow))
            deps = dataflow.collateral_damage("prt2")
            self.assertEquals(deps, set(["prt2"]))
            deps = dataflow.collateral_damage("prt")
            self.assertEquals(deps, set(["prt"]))
            deps = dataflow.collateral_damage("prt", "prt2")
            self.assertEquals(deps, set(["prt", "prt2"]))
            dataflow.delete_modules("prt2")
        self.assertFalse("prt2" in s)
        aio.run(s.step())


if __name__ == "__main__":
    ProgressiveTest.main()
