"""Regression tests for the Tk interpreter-init retry in ``tests/conftest.py``.

The retry exists because repeated Tk() creation/teardown in one pytest process
occasionally fails on Windows with a partial Tcl library init. Every interpreter
creation funnels through ``_tkinter.create``, which conftest wraps. These tests
mock that call so they need no display and no real Tk interpreter.

``conftest`` is importable as a top-level module on both test runners because
pytest/unittest insert the ``tests`` directory on ``sys.path``.
"""

import _tkinter
import tkinter
import unittest

from conftest import _create_tk_interpreter_intercepted


class TestTkInitRetry(unittest.TestCase):
    def setUp(self) -> None:
        self._original_create = _tkinter.create
        self.calls = 0

    def tearDown(self) -> None:
        _tkinter.create = self._original_create

    def _install_failing_create(self, message: str, fail_count: int = 1):
        def fake_create(*args, **kwargs):
            self.calls += 1
            if self.calls <= fail_count:
                raise tkinter.TclError(message)
            return "interp"

        _tkinter.create = fake_create

    def test_missing_tcl_startup_script_is_retried(self):
        # The symptom observed in practice: the pinned library directory is
        # missing a startup script after interpreter churn.
        self._install_failing_create(
            "couldn't read file "
            '"C:/Python/tcl/tcl8.6/auto.tcl": no such file or directory'
        )

        interpreter = _create_tk_interpreter_intercepted()

        self.assertEqual(interpreter, "interp")
        self.assertEqual(self.calls, 2)

    def test_unrelated_read_error_propagates(self):
        # A missing file that is not one of Tcl's startup scripts must not be
        # swallowed by the retry: it is a real, non-retriable failure.
        self._install_failing_create(
            'couldn\'t read file "C:/photos/input.jpg": no such file or directory'
        )

        with self.assertRaises(tkinter.TclError):
            _create_tk_interpreter_intercepted()

        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
