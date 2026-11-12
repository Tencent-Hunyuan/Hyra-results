"""Package version and, for an installed wheel (HYFF_TEST_INSTALLED=1), the distribution metadata."""
import importlib.metadata
import os
import unittest

import hyff


class VersionTest(unittest.TestCase):
    def test_package_version(self):
        self.assertEqual(hyff.__version__, '0.3.0')

    @unittest.skipUnless(os.environ.get('HYFF_TEST_INSTALLED') == '1', 'source checkout')
    def test_installed_distribution(self):
        self.assertEqual(importlib.metadata.version('hyff'), hyff.__version__)
        self.assertNotIn('site-packages', __file__)
        self.assertIn('site-packages', hyff.__file__)


if __name__ == '__main__':
    unittest.main()
