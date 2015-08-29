import unittest

from progressivis import *

class TestProgressiveModule(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def test_scheduler(self):
        self.assertEqual(len(self.scheduler), 0)

    def test_module(self):
        module = Module(id='a', scheduler=self.scheduler)
        self.assertEqual(module.id, 'a')
        self.assertEqual(self.scheduler.exists('a'), True)

if __name__ == '__main__':
    unittest.main()
