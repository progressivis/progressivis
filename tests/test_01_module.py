from progressivis import Module, Every, ProgressiveError, Table

from . import ProgressiveTest
from progressivis.core.module import ReturnRunStep

from typing import Any


class SimpleModule(Module):
    def __init__(self, **kwds: Any):
        super(SimpleModule, self).__init__(**kwds)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        return self._return_run_step(self.state_blocked, 0)


class TestProgressiveModule(ProgressiveTest):
    def test_scheduler(self) -> None:
        self.assertEqual(len(self.scheduler()), 0)

    def test_tags(self) -> None:
        s = self.scheduler()
        simple = SimpleModule(scheduler=s)
        self.assertEquals(simple.tags, set())  # no tags
        with Module.tagged("a", "b"):
            simple2 = SimpleModule(scheduler=s)
        self.assertEquals(simple2.tags, set(["a", "b"]))

    def test_module(self) -> None:
        # pylint: disable=broad-except
        s = self.scheduler()
        with self.assertRaises(TypeError):  # abstract base class
            module = Module(name="a", scheduler=s)  # type: ignore

        with s:
            module = Every(proc=self.terse, name="a", scheduler=s)
            self.assertEqual(module.name, "a")
            self.assertEqual(s.exists("a"), True)
            self.assertEqual(module.get_progress(), (0, 0))
            try:
                module = SimpleModule(name="a", scheduler=s)
                self.fail("Exception not triggered with a duplicate name")
            except ProgressiveError:
                self.assertTrue(True)
            else:
                self.fail("Unexpected exception")
            mod2 = SimpleModule(name="b", scheduler=s)
            self.assertEqual(mod2.get_progress(), (0, 0))
            self.assertTrue(module.is_valid())
            self.assertFalse(module.is_visualization())
            self.assertIsNone(module.get_visualization())
            self.assertIsNone(module.get_data("error"))
            self.assertEquals(module.last_time(), 0)
            module.debug = True
            self.assertEqual(module.params.debug, True)
            module.set_current_params({"quantum": 2.0})
            self.assertEqual(module.params.quantum, 2.0)
            params = module.get_data("_params")
            self.assertIsInstance(params, Table)
            del s["a"]
        self.assertEqual(s.exists("a"), False)
        # module.describe()
        # json = module.to_json(short=True)
        # self.assertEqual(json.get('is_running'), False)
        # self.assertEqual(json.get('is_terminated'), False)
        # json = module.to_json(short=False)
        # self.assertEqual(json.get('start_time', 0), None)
        # maybe check others
        # self.assertFalse(module.has_any_output())


if __name__ == "__main__":
    ProgressiveTest.main()
