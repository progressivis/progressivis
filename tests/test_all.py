import unittest

import test_module
import test_changemanager
import test_merge
import test_csv
import test_vec
import test_stats
import test_percentiles
import test_histogram2d

tests = [test_module.suite,
         test_changemanager.suite,
         test_merge.suite,
         test_csv.suite,
         test_vec.suite,
         test_stats.suite,
         test_percentiles.suite,
         test_histogram2d.suite]
alltests = unittest.TestSuite(tests)
