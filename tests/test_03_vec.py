from . import ProgressiveTest, skip

from progressivis.io import VECLoader
from progressivis.datasets import get_dataset


class TestProgressiveLoadVEC(ProgressiveTest):
    @skip("Need to implement sparse columns")
    def test_read_vec(self) -> None:
        module = VECLoader(get_dataset("warlogs"), name="test_read_vec")
        # self.assertTrue(module.table() is None)
        module.run(0)
        _ = module.trace_stats(max_runs=1)
        df = module.result
        self.assertFalse(df is None)
        _ = len(df)
        # self.assertEqual(l, len(df[df[UPDATE_COLUMN]==module.last_update()]))
        cnt = 1

        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            _ = module.trace_stats(max_runs=1)
            df = module.result
            _ = len(df)
            # print ("Run time: %gs, loaded %d rows" % (s['duration'][len(s)-1], ln))
            # self.assertEqual(ln-l, len(df[df[UPDATE_COLUMN]==module.last_update()]))
            # l = ln
        _ = module.trace_stats(max_runs=1)
        _ = len(module.result)
        # print("Done. Run time: %gs, loaded %d rows" % (s['duration'][len(s)-1], ln))
        # df2 = module.df().groupby([UPDATE_COLUMN])
        # self.assertEqual(cnt, len(df2))


if __name__ == "__main__":
    ProgressiveTest.main()
