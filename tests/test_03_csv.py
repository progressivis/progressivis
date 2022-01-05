from . import ProgressiveTest

from progressivis.core import aio, Sink
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset
from progressivis.core.utils import RandomBytesIO


class TestProgressiveLoadCSV(ProgressiveTest):
    # def setUpNO(self):
    #     self.logger=logging.getLogger('progressivis.core')
    #     self.saved=self.logger.getEffectiveLevel()
    #     self.logger.setLevel(logging.DEBUG)
    #     ch = logging.StreamHandler(stream=sys.stdout)
    #     self.logger.addHandler(ch)

    # def tearDownNO(self):
    #     self.logger.setLevel(self.saved)

    def runit(self, module):
        module.run(1)
        table = module.result
        self.assertFalse(table is None)
        _ = len(table)
        cnt = 2

        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            table = module.result
            _ = len(table)
        _ = module.trace_stats(max_runs=1)
        return cnt

    def test_read_csv(self):
        s = self.scheduler()
        module = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)

    def test_read_fake_csv(self):
        s = self.scheduler()
        module = CSVLoader(
            RandomBytesIO(cols=30, rows=1000000),
            index_col=False,
            header=None,
            scheduler=s,
        )
        self.assertTrue(module.result is None)
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)

    def test_read_multiple_csv(self):
        s = self.scheduler()
        filenames = Table(
            name="file_names",
            dshape="{filename: string}",
            data={"filename": [get_dataset("smallfile"), get_dataset("smallfile")]},
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s)
        csv.input.filenames = cst.output.result
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)

    def test_read_multiple_fake_csv(self):
        s = self.scheduler()
        filenames = Table(
            name="file_names2",
            dshape="{filename: string}",
            data={
                "filename": [
                    "buffer://fake1?cols=10&rows=30000",
                    "buffer://fake2?cols=10&rows=30000",
                ]
            },
        )
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s)
        csv.input.filenames = cst.output.result
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = csv.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)

    def test_as_array(self):
        s = self.scheduler()
        module = CSVLoader(
            get_dataset("bigfile"),
            index_col=False,
            as_array="array",
            header=None,
            scheduler=s,
        )
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        self.assertTrue(module.result is None)
        aio.run(s.start())
        table = module.result
        self.assertEqual(len(table), 1000000)
        self.assertEqual(table.columns, ["array"])
        self.assertEqual(table["array"].shape, (1000000, 30))

    def test_as_array2(self):
        s = self.scheduler()
        module = CSVLoader(
            get_dataset("bigfile"),
            index_col=False,
            as_array={
                "firsthalf": ["_" + str(r) for r in range(13)],
                "secondhalf": ["_" + str(r) for r in range(13, 30)],
            },
            header=None,
            scheduler=s,
        )
        sink = Sink(name="sink", scheduler=s)
        sink.input.inp = module.output.result
        self.assertTrue(module.result is None)
        aio.run(s.start())
        table = module.result
        self.assertEqual(len(table), 1000000)
        self.assertEqual(table.columns, ["firsthalf", "secondhalf"])
        self.assertEqual(table["firsthalf"].shape, (1000000, 13))
        self.assertEqual(table["secondhalf"].shape, (1000000, 17))

    def test_as_array3(self):
        s = self.scheduler()
        try:
            module = CSVLoader(
                get_dataset("mnist_784"),
                index_col=False,
                as_array=lambda cols: {"array": [c for c in cols if c != "class"]},
                scheduler=s,
                )
            sink = Sink(name="sink", scheduler=s)
            sink.input.inp = module.output.result
            self.assertTrue(module.result is None)
            aio.run(s.start())
            table = module.result
            self.assertEqual(len(table), 70000)
            self.assertEqual(table.columns, ["array", "class"])
            self.assertEqual(table["array"].shape, (70000, 784))
            self.assertEqual(table["class"].shape, (70000,))
        except TimeoutError:
            print("Cannot download mnist")
            pass


if __name__ == "__main__":
    ProgressiveTest.main()
