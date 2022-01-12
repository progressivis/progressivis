from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Min, Max, RandomTable
from progressivis.datasets import get_dataset
from progressivis.core import aio, SlotDescriptor, Module, Sink
from progressivis.core.dataflow import Dataflow
from progressivis.vis import MCScatterPlot

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

        table = RandomTable(
            name="table", columns=["a"], throttle=1000, scheduler=scheduler
        )
        m = Min(name="min", scheduler=scheduler)
        prt = Print(proc=self.terse, name="print_min", scheduler=scheduler)
        m.input.table = table.output.result
        prt.input.df = m.output.result
        started = False

        def proc(x):
            nonlocal started
            print("proc max called")
            started = True

        async def _add_max(scheduler, run_number):
            with scheduler:
                print("adding new modules")
                m = Max(name="max", scheduler=scheduler)
                prt = Print(name="print_max", proc=proc, scheduler=scheduler)
                m.input.table = table.output.result
                prt.input.df = m.output.result

        scheduler.on_loop(_add_max, 5)  # run the function after 5 loops
        scheduler.on_loop(self._stop, 10)

        # from nose.tools import set_trace; set_trace()
        aio.run(scheduler.start())
        self.assertTrue(started)

    def test_dataflow_2_add_remove(self):
        scheduler = self.scheduler(clean=True)

        table = RandomTable(
            name="table", columns=["a"], throttle=1000, scheduler=scheduler
        )
        m = Min(name="min", scheduler=scheduler)
        prt = Print(proc=self.terse, name="print_min", scheduler=scheduler)
        m.input.table = table.output.result
        prt.input.df = m.output.result
        started = False

        def proc(x):
            nonlocal started
            print("proc max called")
            started = True

        async def _add_max_remove_min(scheduler, run_number):
            with scheduler as dataflow:
                print("adding new modules")
                m = Max(name="max", scheduler=scheduler)
                prt = Print(name="print_max", proc=proc, scheduler=scheduler)
                m.input.table = table.output.result
                prt.input.df = m.output.result
                print("removing min module")
                dataflow.delete_modules("min", "print_min")

        # t = _add_max_remove_min(csv, scheduler, proc=proc)
        scheduler.on_loop(_add_max_remove_min, 5)
        scheduler.on_loop(self._stop, 10)
        aio.run(scheduler.start())
        self.assertTrue(started)

    def test_dataflow_3_dels(self):
        s = self.scheduler()
        table = RandomTable(name="table", columns=["a"], throttle=1000, scheduler=s)
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
        table = RandomTable(name="table", columns=["a"], throttle=1000, scheduler=s)
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
        table = RandomTable(name="table", columns=["a"], throttle=1000, scheduler=s)
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
        table = RandomTable(name="table", columns=["a"], throttle=1000, scheduler=s)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = table.output.result
        prt = Print(name="prt", proc=self.terse, scheduler=s)
        prt.input.df = table.output.result
        prt2 = Print(name="prt2", proc=self.terse, scheduler=s)
        prt2.input.df = table.output.result
        s.commit()

        async def modify_1(scheduler, run_number):
            with s as dataflow:
                print("Checking module deletion")
                self.assertTrue(isinstance(dataflow, Dataflow))
                deps = dataflow.collateral_damage("prt2")
                self.assertEquals(deps, set(["prt2"]))
                deps = dataflow.collateral_damage("prt")
                self.assertEquals(deps, set(["prt"]))
                deps = dataflow.collateral_damage("prt", "prt2")
                self.assertEquals(deps, set(["prt", "prt2"]))
                dataflow.delete_modules("prt2")
            s.on_loop(modify_2, 5)

        async def modify_2(scheduler, run_number):
            self.assertFalse("prt2" in scheduler)
            with s as dataflow:
                print("Checking more module deletion")
                deps = dataflow.collateral_damage("prt")
                self.assertEquals(deps, {"prt"})
                deps = dataflow.collateral_damage("prt", "sink")
                self.assertEquals(deps, {"prt", "sink", "table"})
                dataflow.delete_modules("prt")
            s.on_loop(modify_3, 5)

        async def modify_3(scheduler, run_number):
            self.assertFalse("prt" in scheduler)
            with s as dataflow:
                print("Checking even more module deletion")
                deps = dataflow.collateral_damage("sink")
                self.assertEquals(deps, {"sink", "table"})
                # from nose.tools import set_trace; set_trace()
                dataflow.delete_modules("sink", "table")

        async def stop_error(scheduler, run_number):
            self.assertFalse("Scheduler should have stopped")
            await self._stop()

        s.on_loop(modify_1, 5)
        s.on_loop(stop_error, 100)
        aio.run(s.start())
        # from nose.tools import set_trace; set_trace()

    def test_dataflow_7_dynamic(self):
        s = self.scheduler()
        table = RandomTable(
            name="table", columns=["a", "b", "c"], throttle=1000, scheduler=s
        )
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = table.output.result
        s.commit()

        # Start loading a dataset, then visualize it, then change the visualizations

        async def modify_1(scheduler, run_number):
            print("Adding scatterplot_1")
            # from nose.tools import set_trace; set_trace()
            with scheduler as dataflow:
                sp = MCScatterPlot(
                    name="scatterplot_1",
                    classes=[("Scatterplot", "a", "b")],
                    approximate=True,
                    scheduler=scheduler,
                )
                sp.create_dependent_modules(table, "result")
                print(f"Created scatterplot_1, groups: {dataflow.groups()}")
            scheduler.on_loop(modify_2, 10)  # Schedule the next activity

        async def modify_2(scheduler, run_number):
            print("Removing scatterplot_1")
            self.assertTrue("scatterplot_1" in scheduler)
            with scheduler as dataflow:
                print("Checking scatterplot_1 module deletion")
                modules = sorted(dataflow.group_modules("scatterplot_1"))
                print(f"Removing modules {modules}")
                deps = dataflow.collateral_damage(*modules)
                print(f"collateral_damage('scatterplot_1') = '{sorted(deps)}'")
                dataflow.delete_modules(*deps)
            scheduler.on_loop(modify_3, 10)

        async def modify_3(scheduler, run_number):
            print("Adding scatterplot_2")
            self.assertFalse("scatterplot_1" in scheduler)
            with scheduler:
                sp = MCScatterPlot(
                    name="scatterplot_2",
                    classes=[("Scatterplot", "a", "c")],
                    approximate=True,
                    scheduler=scheduler,
                )
                sp.create_dependent_modules(table, "result")
            scheduler.on_loop(modify_4, 10)  # Schedule the next activity

        async def modify_4(scheduler, run_number):
            print("Removing scatterplot_2")
            self.assertFalse("scatterplot_1" in scheduler)
            self.assertTrue("scatterplot_2" in scheduler)
            with scheduler as dataflow:
                print("Checking scatterplot_2 module deletion")
                print("Checking scatterplot module deletion")
                modules = sorted(dataflow.group_modules("scatterplot_2"))
                print(f"Removing modules {modules}")
                deps = dataflow.collateral_damage(*modules)
                print(f"collateral_damage('scatterplot_2') = '{sorted(deps)}'")
                dataflow.delete_modules(*deps)
            s.on_loop(modify_5, 5)

        async def modify_5(scheduler, run_number):
            print("Removing table")
            self.assertFalse("scatterplot_1" in scheduler)
            self.assertFalse("scatterplot_2" in scheduler)
            with scheduler as dataflow:
                print("Checking sink+table modules deletion")
                deps = dataflow.collateral_damage("sink")
                print(f"collateral_damage('sink') = '{sorted(deps)}'")
                dataflow.delete_modules(*deps)

        async def stop_error(scheduler, run_number):
            self.assertFalse("Scheduler should have stopped")
            await self._stop()

        s.on_loop(modify_1, 10)
        s.on_loop(stop_error, 100)
        aio.run(s.start())
        self.assertFalse("scatterplot_1" in s)
        self.assertFalse("scatterplot_2" in s)
        # from nose.tools import set_trace; set_trace()


if __name__ == "__main__":
    ProgressiveTest.main()
