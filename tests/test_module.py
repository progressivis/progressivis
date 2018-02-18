from progressivis import Module, Every, ProgressiveError, Table

from . import ProgressiveTest


class SimpleModule(Module):
    def run_step(self, run_number, step_size, howlong):  # pragma no cover
        return self._return_run_step(self.state_blocked, 0)

class TestProgressiveModule(ProgressiveTest):
    def test_scheduler(self):
        self.assertEqual(len(self.scheduler()), 0)

    def test_module(self):
        # pylint: disable=broad-except
        s = self.scheduler()
        with self.assertRaises(TypeError):  # abstract base class
            module = Module(mid='a', scheduler=s)
        with self.assertRaises(ValueError): # name and mid specified
            module = SimpleModule(mid='a', name='foo', scheduler=s)

        module = Every(proc=self.terse, mid='a', scheduler=s)
        self.assertEqual(module.id, 'a')
        self.assertEqual(s.exists('a'), True)
        self.assertEqual(module.get_progress(), (0, 0))
        with self.assertRaises(ProgressiveError):
            module = SimpleModule(mid='a', scheduler=s)
        mod2 = SimpleModule(mid='b', scheduler=s)
        self.assertEqual(mod2.get_progress(), (0, 0))
        module.debug = True
        self.assertEqual(module.params.debug, True)
        module.set_current_params({'quantum': 2.0})
        self.assertEqual(module.params.quantum, 2.0)
        params = module.get_data("_params")
        self.assertIsInstance(params, Table)
        module.destroy()
        self.assertEqual(s.exists('a'), False)
        module.describe()

if __name__ == '__main__':
    ProgressiveTest.main()
