"""Regression for the direct algebra(...) MCP/CLI query constructor."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
spec = importlib.util.spec_from_file_location("algebra_plugin_transport", MODULE / "plugin.py")
assert spec and spec.loader
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class TransportRegistrationTests(unittest.TestCase):
    def test_enabled_module_claims_algebra_query_prefix(self):
        claimed: list[str] = []
        flex_module = types.ModuleType("flex")
        core_module = types.ModuleType("flex.mcp_core")
        core_module.register_query_prefix = claimed.append
        flex_module.mcp_core = core_module

        with (
            patch.dict(os.environ, {"FLEX_ALGEBRA": "1"}),
            patch.dict(
                sys.modules,
                {"flex": flex_module, "flex.mcp_core": core_module},
            ),
        ):
            materializers = plugin.register_query_materializers()

        self.assertEqual(claimed, ["algebra("])
        self.assertEqual(materializers, [plugin.algebra_materializer])

    def test_disabled_module_does_not_claim_prefix(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(plugin.register_query_materializers(), [])


if __name__ == "__main__":
    unittest.main()
