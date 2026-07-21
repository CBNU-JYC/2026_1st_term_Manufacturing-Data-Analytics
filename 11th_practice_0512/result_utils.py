"""
Result saving helpers.

All scripts in this folder write terminal outputs to ./0_result so that
code, data, and generated results stay separated.
"""

from __future__ import annotations

from pathlib import Path
import atexit
import sys


RESULT_DIR = Path(__file__).resolve().parent / "0_result"


def get_result_dir() -> Path:
    """Create and return this folder's result directory."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR


class _Tee:
    """Write the same terminal stream to the screen and to a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def start_terminal_log(filename: str) -> Path:
    """
    Save the terminal output of the current script to ./0_result/<filename>.

    The output is still shown in the terminal while an identical text copy is
    written to the result file for easy review.
    """
    path = get_result_dir() / filename
    log_file = path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = _Tee(original_stdout, log_file)
    stderr_tee = _Tee(original_stderr, log_file)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def restore_streams():
        if sys.stdout is stdout_tee:
            sys.stdout = original_stdout
        if sys.stderr is stderr_tee:
            sys.stderr = original_stderr
        log_file.close()

    atexit.register(restore_streams)
    return path
