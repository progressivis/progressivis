from . import ProgressiveTest
from progressivis.core import aio
from progressivis.datasets import get_dataset
from progressivis.core import Scheduler

import progressivis.expr as pv
from progressivis.expr.table import PipedInput


def prtm(x):
    print("m: ", len(x))


def prtM(x):
    print("M: ", len(x))


def prtT(x):
    print("T: ", len(x))


# def prtrepr(x):
#    print(repr(x))


class TestExpr(ProgressiveTest):
    def setUp(self):
        super(TestExpr, self).setUp()
        Scheduler.default = self.scheduler()

    def test_load_csv(self):
        """
        Connecting modules via function calls
        """
        with Scheduler.default:
            csv = pv.load_csv(get_dataset("bigfile"), index_col=False, header=None)
            m = pv.min(csv)
            pv.echo(m, proc=prtm)
            M = pv.max(csv)
            pv.echo(M, proc=prtM)
            trace = M["_trace"]
            pv.echo(trace, proc=prtT)
            self.assertEqual(csv.scheduler(), csv.module.scheduler())
        aio.run(csv.scheduler().start())
        table = csv.table
        lastm = m.table
        lastM = M.table
        self.assertEqual(len(table), 1000000)
        for col in table.columns:
            c = table[col]
            v = c.min()
            self.assertEqual(v, lastm[col])
            v = c.max()
            self.assertEqual(v, lastM[col])

    def test_piped_load_csv(self):
        """
        Connecting modules via the pipe operator ( 3 pipes)
        """
        with Scheduler.default:
            ret = (
                PipedInput(get_dataset("bigfile"))
                | pv.load_csv(index_col=False, header=None)
                | pv.min()
                | pv.echo(proc=prtm)
            )
            csv = ret.repipe("csv_loader_1")
            _ = csv | pv.max() | pv.echo(proc=prtM)
            m = ret.fetch("min_1")
            M = ret.fetch("max_1")
            _ = M["_trace"] | pv.echo(proc=prtT)
            self.assertEqual(csv.scheduler(), csv.module.scheduler())
        aio.run(csv.scheduler().start())
        table = csv.table
        lastm = m.table
        lastM = M.table
        self.assertEqual(len(table), 1000000)
        for col in table.columns:
            # print('testing column %s'%col)
            c = table[col]
            v = c.min()
            self.assertEqual(v, lastm[col])
            v = c.max()
            self.assertEqual(v, lastM[col])

    def test_piped_load_csv2(self):
        """
        Connecting modules via the pipe operator (only one pipe)
        """
        with Scheduler.default:
            ret = (
                PipedInput(get_dataset("bigfile"))
                | pv.load_csv(index_col=False, header=None)
                | pv.min()
                | pv.echo(proc=prtm).repipe("csv_loader_1")
                | pv.max()
                | pv.echo(proc=prtM).repipe("max_1", out="_trace")
                | pv.echo(proc=prtT)
            )
            m = ret.fetch("min_1")
            M = ret.fetch("max_1")
            csv = ret.fetch("csv_loader_1")
            self.assertEqual(csv.scheduler(), csv.module.scheduler())
        aio.run(csv.scheduler().start())
        table = csv.table
        lastm = m.table
        lastM = M.table
        self.assertEqual(len(table), 1000000)
        for col in table.columns:
            # print('testing column %s'%col)
            c = table[col]
            v = c.min()
            self.assertEqual(v, lastm[col])
            v = c.max()
            self.assertEqual(v, lastM[col])
