"""Test command-line interface for MaxLEV."""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.cli import parse_args, main


class TestParseArgs:
    """Tests for parse_args function."""

    def test_config_required(self):
        """Test that config argument is required."""
        with patch('sys.argv', ['maxlev']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_config_parsed(self):
        """Test that config path is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json']):
            args = parse_args()
            assert args.config == 'config.json'

    def test_validate_flag(self):
        """Test --validate flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--validate']):
            args = parse_args()
            assert args.validate is True

    def test_verbose_flag(self):
        """Test --verbose flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--verbose']):
            args = parse_args()
            assert args.verbose is True

    def test_verbose_short_flag(self):
        """Test -v flag is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '-v']):
            args = parse_args()
            assert args.verbose is True

    def test_seed_option(self):
        """Test --seed option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--seed', '42']):
            args = parse_args()
            assert args.seed == 42

    def test_maxiter_option(self):
        """Test --maxiter option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--maxiter', '100']):
            args = parse_args()
            assert args.maxiter == 100

    def test_output_dir_option(self):
        """Test --output-dir option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--output-dir', '/tmp/out']):
            args = parse_args()
            assert args.output_dir == '/tmp/out'

    def test_workers_option(self):
        """Test --workers option is parsed."""
        with patch('sys.argv', ['maxlev', 'config.json', '--workers', '4']):
            args = parse_args()
            assert args.workers == 4

    def test_default_values(self):
        """Test default values for optional arguments."""
        with patch('sys.argv', ['maxlev', 'config.json']):
            args = parse_args()
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
            args = parse_args()
            assert args.seed == 99
            assert args.maxiter == 500
            assert args.workers == 4
            assert args.verbose is True

    def test_output_dir_override(self):
        """Test that --output-dir is parsed correctly."""
        with patch('sys.argv', ['maxlev', 'config.json', '--output-dir', '/custom/path']):
            args = parse_args()
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
