import os
import unittest
from unittest.mock import patch

from hyqec_check.engines import environment


class EnvironmentCompatibilityTests(unittest.TestCase):
    def test_renamed_settings_override_deprecated_aliases(self):
        for suffix in ("NATIVE_ENGINE", "WIDE_ENGINE", "REFERENCE_ENGINE", "MODULE"):
            with self.subTest(suffix=suffix), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(environment(suffix, "default"), "default")
                os.environ[f"BNBCHECK_{suffix}"] = "legacy"
                self.assertEqual(environment(suffix, "default"), "legacy")
                os.environ[f"HYQEC_CHECK_{suffix}"] = "current"
                self.assertEqual(environment(suffix, "default"), "current")
