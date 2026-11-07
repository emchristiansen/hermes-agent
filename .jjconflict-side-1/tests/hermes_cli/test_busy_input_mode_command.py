"""Tests for the /busy CLI command and busy-input-mode config handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod

class TestHandleBusyCommand(unittest.TestCase):
    def _make_cli(self, busy_input_mode="interrupt"):
        return SimpleNamespace(
            busy_input_mode=busy_input_mode,
            agent=None,
        )

    def test_no_args_shows_status(self):
        cli_mod = _import_cli()
        stub = self._make_cli("queue")
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("queue", printed)
        self.assertIn("interrupt", printed)

    def test_every_mode_argument_reports_steer_and_saves_nothing(self):
        """Every message steers (``CLIInitMixin.busy_input_mode``): /busy only reports it."""
        cli_mod = _import_cli()
        for arg in ("queue", "interrupt", "steer"):
            stub = self._make_cli("steer")
            with (
                patch.object(cli_mod, "_cprint") as mock_cprint,
                patch.object(cli_mod, "save_config_value", return_value=True) as mock_save,
            ):
                cli_mod.HermesCLI._handle_busy_command(stub, f"/busy {arg}")
            self.assertEqual(stub.busy_input_mode, "steer")
            mock_save.assert_not_called()
            printed = " ".join(str(c) for c in mock_cprint.call_args_list)
            self.assertIn("steer", printed.lower())

    def test_invalid_argument_prints_usage(self):
        cli_mod = _import_cli()
        stub = self._make_cli()
        with (
            patch.object(cli_mod, "_cprint") as mock_cprint,
            patch.object(cli_mod, "save_config_value") as mock_save,
        ):
            cli_mod.HermesCLI._handle_busy_command(stub, "/busy nonsense")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /busy", printed)
