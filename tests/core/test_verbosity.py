"""Tests for the opt-in verbosity API and its zero-import-side-effects contract (CPU-only, no models)."""
import logging
import subprocess
import sys

import pytest

from aisteer360.utils import verbosity

PACKAGE_LOGGER = "aisteer360"


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Snapshot and restore the package logger and the env-default guard around each test."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_guard = verbosity._env_default_applied
    try:
        yield
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)
        verbosity._env_default_applied = saved_guard


class TestImportSideEffects:

    def test_import_attaches_only_null_handler_and_leaves_root_untouched(self):
        # a fresh interpreter isolates the one-time import-time effects
        script = (
            "import logging\n"
            "root = logging.getLogger()\n"
            "root_handlers_before = list(root.handlers)\n"
            "root_level_before = root.level\n"
            "import aisteer360\n"
            "pkg = logging.getLogger('aisteer360')\n"
            "non_null = [h for h in pkg.handlers if not isinstance(h, logging.NullHandler)]\n"
            "assert pkg.handlers, 'expected a NullHandler on the package logger'\n"
            "assert not non_null, f'unexpected non-null handlers: {non_null}'\n"
            "assert list(root.handlers) == root_handlers_before, 'root handlers changed on import'\n"
            "assert root.level == root_level_before, 'root level changed on import'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


class TestSetVerbosity:

    def test_debug_emits_through_one_attached_handler(self, caplog):
        verbosity.set_verbosity("debug")
        assert verbosity.get_verbosity() == logging.DEBUG

        module_logger = logging.getLogger("aisteer360.some.module")
        with caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER):
            module_logger.debug("hello from a toolkit module")
        assert any("hello from a toolkit module" in record.message for record in caplog.records)

    def test_repeated_calls_do_not_attach_a_second_handler(self):
        logger = logging.getLogger(PACKAGE_LOGGER)
        real_before = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]

        verbosity.set_verbosity("info")
        after_first = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
        verbosity.set_verbosity("debug")
        after_second = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]

        assert len(after_first) == len(real_before) + 1
        assert len(after_second) == len(after_first)  # idempotent handler attachment
        assert logger.level == logging.DEBUG  # level still updates on the second call

    def test_accepts_logging_constant(self):
        verbosity.set_verbosity(logging.WARNING)
        assert logging.getLogger(PACKAGE_LOGGER).level == logging.WARNING

    def test_rejects_unknown_level_name(self):
        with pytest.raises(ValueError, match="Unknown verbosity level"):
            verbosity.set_verbosity("chatty")


class TestEnvDefault:

    def test_env_variable_is_honored(self, monkeypatch):
        logger = logging.getLogger(PACKAGE_LOGGER)
        logger.setLevel(logging.NOTSET)
        verbosity._env_default_applied = False
        monkeypatch.setenv("AISTEER_VERBOSITY", "info")

        level = verbosity.get_verbosity()

        assert level == logging.INFO
        assert logger.level == logging.INFO

    def test_absent_env_variable_leaves_level_untouched(self, monkeypatch):
        logger = logging.getLogger(PACKAGE_LOGGER)
        logger.setLevel(logging.NOTSET)
        verbosity._env_default_applied = False
        monkeypatch.delenv("AISTEER_VERBOSITY", raising=False)

        verbosity.get_verbosity()

        assert logger.level == logging.NOTSET  # default stays silent

    def test_unrecognized_env_value_is_ignored(self, monkeypatch):
        logger = logging.getLogger(PACKAGE_LOGGER)
        logger.setLevel(logging.NOTSET)
        verbosity._env_default_applied = False
        monkeypatch.setenv("AISTEER_VERBOSITY", "nonsense")

        verbosity.get_verbosity()

        assert logger.level == logging.NOTSET


class TestQuietThirdParty:

    def test_runs_without_error_when_transformers_importable(self):
        pytest.importorskip("transformers")
        verbosity.quiet_third_party()  # no raise

    def test_no_op_when_optional_dependency_absent(self, monkeypatch):
        real_import = __import__

        def _fail_hub(name, *args, **kwargs):
            if name.startswith("huggingface_hub"):
                raise ImportError("simulated missing huggingface_hub")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fail_hub)
        verbosity.quiet_third_party()  # the guarded block swallows the ImportError, no raise
