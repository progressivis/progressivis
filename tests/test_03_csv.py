from . import ProgressiveTest
from progressivis.core import aio
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset#, RandomBytesIO
from progressivis.core.utils import RandomBytesIO
#import logging, sys

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
        l = len(table)
        cnt = 2

        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            #s = module.trace_stats(max_runs=1)
            table = module.result
            ln = len(table)
            #print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            #self.assertEqual(ln-l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        #print("Done. Run time: %gs, loaded %d rows" % (s['duration'][-1], len(module.result)))
        return cnt

    def test_read_csv(self):
        s=self.scheduler()
        module=CSVLoader(get_dataset('bigfile'), index_col=False, header=None, scheduler=s)
        self.assertTrue(module.result is None)
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)



    def test_read_fake_csv(self):
        s=self.scheduler()
        module=CSVLoader(RandomBytesIO(cols=30, rows=1000000), index_col=False, header=None, scheduler=s)
        self.assertTrue(module.result is None)
        aio.run(s.start())
        self.assertEqual(len(module.result), 1000000)

    def test_read_multiple_csv(self):
        s=self.scheduler()
        filenames = Table(name='file_names',
                          dshape='{filename: string}',
                          data={'filename': [get_dataset('smallfile'), get_dataset('smallfile')]})
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s)
        csv.input.filenames = cst.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)

    def test_read_multiple_fake_csv(self):
        s=self.scheduler()
        filenames = Table(name='file_names2',
                          dshape='{filename: string}',
                          data={'filename': [
                              'buffer://fake1?cols=10&rows=30000',
                              'buffer://fake2?cols=10&rows=30000']})
        cst = Constant(table=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s)
        csv.input.filenames = cst.output.result
        aio.run(csv.start())
        self.assertEqual(len(csv.result), 60000)



if __name__ == '__main__':
    ProgressiveTest.main()
