"""Test command-line interface for MaxLEV."""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.cli import fParseArgs, main


class TestParseArgs:
    """Tests for fParseArgs function."""

    def test_config_required(self):
        """Test that config argument is required."""
        with patch('sys.argv', ['maxlev']):
            with pytest.raises(SystemExit):
                fParseArgs()

    def test_config_parsed(self):
        """Test that config path is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json']):
            args = fParseArgs()
            assert args.config == 'config.json'

    def test_validate_flag(self):
        """Test --validate flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--validate']):
            args = fParseArgs()
            assert args.validate is True

    def test_verbose_flag(self):
        """Test --verbose flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--verbose']):
            args = fParseArgs()
            assert args.verbose is True

    def test_verbose_short_flag(self):
        """Test -v flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '-v']):
            args = fParseArgs()
            assert args.verbose is True

    def test_seed_option(self):
        """Test --seed option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--seed', '42']):
            args = fParseArgs()
            assert args.seed == 42

    def test_maxiter_option(self):
        """Test --maxiter option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--maxiter', '100']):
            args = fParseArgs()
            assert args.maxiter == 100

    def test_output_dir_option(self):
        """Test --output-dir option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--output-dir', '/tmp/out']):
            args = fParseArgs()
            assert args.output_dir == '/tmp/out'

    def test_workers_option(self):
        """Test --workers option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--workers', '4']):
            args = fParseArgs()
            assert args.workers == 4

    def test_default_values(self):
        """Test default values for optional arguments."""
        with patch('sys.argv', ['maxlev', 'config.json']):
            args = fParseArgs()
            assert args.validate is False
            assert args.verbose is False
            assert args.seed is None
            assert args.maxiter is None
            assert args.output_dir is None
            assert args.workers is None


class TestMainValidation:
    """Tests for main function - validation paths."""

    def test_exits_on_missing_config(self):
        """Test that main exits when config file not found."""
        with patch('sys.argv', ['maxlev', 'nonexistent.json']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_exits_on_non_json_config(self):
        """Test that main exits for non-JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not json")
            config_path = f.name

        try:
            with patch('sys.argv', ['maxlev', config_path]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1
        finally:
            Path(config_path).unlink()

    def test_exits_on_validation_errors(self):
        """Test that main exits when config has validation errors."""
        config_data = {
            'name': 'test',
            'parameters': [],  # No parameters -> validation error
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('sys.argv', ['maxlev', config_path]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1
        finally:
            Path(config_path).unlink()

    def test_validate_mode_exits_cleanly(self):
        """Test that --validate mode exits cleanly on valid config."""
        config_data = {
            'name': 'test',
            'vplanet': {'inpath': '/tmp', 'executable': 'echo'},
            'parameters': [
                {'name': 'star.dMass', 'bounds': [0.5, 1.5], 'units': 'Msun'}
            ],
            'outputs': [
                {'name': 'final.star.Mass', 'units': 'Msun'}
            ],
            'observables': [
                {
                    'name': 'Mass',
                    'type': 'direct',
                    'output': 'final.star.Mass',
                    'observed_value': 1.0,
                    'uncertainty': 0.1
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('sys.argv', ['maxlev', config_path, '--validate']):
                with patch('maxlev.config.validation.validate_all', return_value=[]):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0
        finally:
            Path(config_path).unlink()


class TestMainOverrides:
    """Tests for main function - command line overrides."""

    def test_args_are_parsed_correctly(self):
        """Test that command line arguments are parsed correctly."""
        with patch('sys.argv', ['maxlev', 'config.json', '--seed', '99',
                                '--maxiter', '500', '--workers', '4', '-v']):
            args = fParseArgs()
            assert args.seed == 99
            assert args.maxiter == 500
            assert args.workers == 4
            assert args.verbose is True

    def test_output_dir_override(self):
        """Test that --output-dir is parsed correctly."""
        with patch('sys.argv', ['maxlev', 'config.json', '--output-dir', '/custom/path']):
            args = fParseArgs()
            assert args.output_dir == '/custom/path'

    def test_error_formatting_with_line_number(self):
        """Test that errors with line numbers are formatted correctly."""
        config_data = {
            'name': 'test',
            'parameters': [],
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('sys.argv', ['maxlev', config_path]):
                with patch('maxlev.config.validation.validate_all',
                          return_value=[('Error with line', 42)]):
                    with patch('builtins.print') as mock_print:
                        with pytest.raises(SystemExit):
                            main()
                        calls = [str(c) for c in mock_print.call_args_list]
                        assert any('Line 42' in str(c) for c in calls)
        finally:
            Path(config_path).unlink()

    def test_error_formatting_without_line_number(self):
        """Test that errors without line numbers are formatted correctly."""
        config_data = {
            'name': 'test',
            'parameters': [],
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('sys.argv', ['maxlev', config_path]):
                with patch('maxlev.config.validation.validate_all',
                          return_value=[('Error without line', 0)]):
                    with patch('builtins.print') as mock_print:
                        with pytest.raises(SystemExit):
                            main()
                        calls = [str(c) for c in mock_print.call_args_list]
                        assert any('Error without line' in str(c) for c in calls)
        finally:
            Path(config_path).unlink()


class TestFailureReporting:
    """Tests for CLI failure reporting."""

    def test_reports_failed_optimization(self, capsys):
        """Test that penalty-value result is reported as failure."""
        from maxlev.cli import _fnReportFailedOptimization

        mock_model = MagicMock()
        mock_model.iSimulationCount = 50
        mock_model.iSimulationFailureCount = 50

        _fnReportFailedOptimization(1e10, mock_model)

        captured = capsys.readouterr()
        assert "OPTIMIZATION FAILED" in captured.out
        assert "50" in captured.out

    def test_reports_all_simulations_failed(self, capsys):
        """Test that AllSimulationsFailedError message is informative."""
        from maxlev.model import AllSimulationsFailedError

        error = AllSimulationsFailedError(
            "All 10 VPLanet simulations failed."
        )
        assert "All 10" in str(error)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestResumeOption:
    """Tests for the --resume option and checkpoint settings."""

    @staticmethod
    def fnApplyCheckpointArguments(listArguments, dictOptimizer):
        """Parse arguments and apply checkpoint settings to a mock config."""
        from maxlev.cli import _fnApplyCheckpointSettings
        config = MagicMock()
        config.name = 'venusFit'
        config.optimizer = dictOptimizer
        with patch('sys.argv', ['maxlev', 'config.json'] + listArguments):
            _fnApplyCheckpointSettings(fParseArgs(), config)
        return config.optimizer

    def test_resume_defaults_to_none(self):
        """Test that --resume is off unless given."""
        with patch('sys.argv', ['maxlev', 'config.json']):
            assert fParseArgs().resume is None

    def test_resume_without_path(self):
        """Test that a bare --resume parses to an empty path."""
        with patch('sys.argv', ['maxlev', 'config.json', '--resume']):
            assert fParseArgs().resume == ''

    def test_resume_with_path(self):
        """Test that --resume accepts a checkpoint path."""
        with patch('sys.argv', ['maxlev', 'config.json', '--resume', 'a.json']):
            assert fParseArgs().resume == 'a.json'

    def test_default_checkpoint_named_after_config(self):
        """Test that the checkpoint defaults to <name>_checkpoint.json."""
        dictOptimizer = self.fnApplyCheckpointArguments([], {})
        assert dictOptimizer['checkpoint_file'] == 'venusFit_checkpoint.json'
        assert 'resume_file' not in dictOptimizer

    def test_configured_checkpoint_kept(self):
        """Test that a configured checkpoint_file is not replaced."""
        dictOptimizer = self.fnApplyCheckpointArguments(
            [], {'checkpoint_file': 'mine.json'})
        assert dictOptimizer['checkpoint_file'] == 'mine.json'

    def test_bare_resume_uses_checkpoint_file(self):
        """Test that --resume alone reads the configured checkpoint."""
        dictOptimizer = self.fnApplyCheckpointArguments(['--resume'], {})
        assert dictOptimizer['resume_file'] == 'venusFit_checkpoint.json'

    def test_resume_path_overrides_checkpoint_file(self):
        """Test that an explicit --resume path is used."""
        dictOptimizer = self.fnApplyCheckpointArguments(
            ['--resume', 'old.json'], {})
        assert dictOptimizer['resume_file'] == 'old.json'
        assert dictOptimizer['checkpoint_file'] == 'venusFit_checkpoint.json'

    def test_bare_resume_with_disabled_checkpoint_exits(self):
        """Test a clear exit when there is no checkpoint to resume."""
        with pytest.raises(SystemExit) as exc_info:
            self.fnApplyCheckpointArguments(
                ['--resume'], {'checkpoint_file': None})
        assert exc_info.value.code == 1

    def test_checkpoint_error_exits_cleanly(self, capsys):
        """Test that a checkpoint problem ends the run with a message."""
        from maxlev.checkpoint import CheckpointError
        from maxlev.cli import _fnRunOptimization
        with patch('maxlev.cli._fInitializeModel',
                   return_value=(MagicMock(), MagicMock())), \
                patch('maxlev.optimizer.Optimizer.optimize',
                      side_effect=CheckpointError('stale checkpoint')):
            with pytest.raises(SystemExit) as exc_info:
                _fnRunOptimization(MagicMock())
        assert exc_info.value.code == 1
        assert 'stale checkpoint' in capsys.readouterr().out
