from . import ProgressiveTest

from progressivis import Module, Every

class TestProgressiveModule(ProgressiveTest):
    def setUp(self):
        super(TestProgressiveModule, self).setUp()

    def test_scheduler(self):
        self.assertEqual(len(self.scheduler()), 0)

    def test_module(self):
        # pylint: disable=broad-except
        s = self.scheduler()
        try:
            module = Module(mid='a', scheduler=s)
        except TypeError:
            pass
        except Exception as e:
            self.fail('Unexpected error while creating abstract module %s'% e)
        else:
            self.fail('Abstract module created, error')
        module = Every(proc=self.terse, mid='a', scheduler=s)
        self.assertEqual(module.id, 'a')
        self.assertEqual(s.exists('a'), True)

if __name__ == '__main__':
    ProgressiveTest.main()
