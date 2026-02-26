"""Test output generation for MaxLEV."""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.output import (
    fnGenerateMaxLEVInputFiles,
    _flistUpdateInputFile,
    _fsUpdateLine,
    _fsFormatValue,
    fnSaveResults,
    fnPlotEvolution,
)
from maxlev.config import ParameterConfig, OutputConfig, ObservableConfig


class TestFormatValue:
    """Tests for _fsFormatValue helper."""

    def test_large_value_scientific_notation(self):
        """Test that large values use scientific notation."""
        result = _fsFormatValue(1e25)
        assert 'e' in result.lower()

    def test_small_value_scientific_notation(self):
        """Test that small values use scientific notation."""
        result = _fsFormatValue(1e-5)
        assert 'e' in result.lower()

    def test_normal_value_decimal(self):
        """Test that normal values use decimal notation."""
        result = _fsFormatValue(3.5)
        assert result == '3.5'

    def test_zero_value(self):
        """Test that zero is formatted as '0'."""
        result = _fsFormatValue(0.0)
        assert result == '0'


class TestUpdateLine:
    """Tests for _fsUpdateLine function."""

    def test_empty_line_unchanged(self):
        """Test that empty lines are unchanged."""
        result = _fsUpdateLine('', {'dTMan': 3000.0})
        assert result == ''

    def test_whitespace_only_unchanged(self):
        """Test that whitespace-only lines are unchanged."""
        result = _fsUpdateLine('   \n', {'dTMan': 3000.0})
        assert result == '   \n'

    def test_comment_line_unchanged(self):
        """Test that comment lines are unchanged."""
        result = _fsUpdateLine('# This is a comment\n', {'dTMan': 3000.0})
        assert result == '# This is a comment\n'

    def test_unmatched_parameter_unchanged(self):
        """Test that lines with unmatched parameters are unchanged."""
        result = _fsUpdateLine('dOtherParam 100\n', {'dTMan': 3000.0})
        assert result == 'dOtherParam 100\n'

    def test_matched_parameter_updated(self):
        """Test that matched parameters get updated values."""
        result = _fsUpdateLine('dTMan 2500\n', {'dTMan': 3000.0})
        assert 'dTMan' in result
        assert '3000' in result

    def test_preserves_leading_whitespace(self):
        """Test that leading whitespace is preserved."""
        result = _fsUpdateLine('  dTMan 2500\n', {'dTMan': 3000.0})
        assert result.startswith('  dTMan')

    def test_preserves_trailing_comment(self):
        """Test that trailing comments are preserved."""
        result = _fsUpdateLine('dTMan 2500 # Initial temp\n', {'dTMan': 3000.0})
        assert '# Initial temp' in result

    def test_scientific_notation_for_large_values(self):
        """Test that large values use scientific notation."""
        result = _fsUpdateLine('dMass 1.0\n', {'dMass': 1e25})
        assert 'e' in result.lower() or 'E' in result

    def test_scientific_notation_for_small_values(self):
        """Test that small values use scientific notation."""
        result = _fsUpdateLine('dTiny 1.0\n', {'dTiny': 1e-5})
        assert 'e' in result.lower() or 'E' in result

    def test_no_trailing_zeros_for_normal_values(self):
        """Test that normal values don't have unnecessary trailing zeros."""
        result = _fsUpdateLine('dValue 1.0\n', {'dValue': 3.5})
        assert '3.5' in result
        assert '3.500' not in result

    def test_single_part_line_unchanged(self):
        """Test that lines with only one part are unchanged."""
        result = _fsUpdateLine('dTMan\n', {'dTMan': 3000.0})
        assert result == 'dTMan\n'


class TestUpdateInputFile:
    """Tests for _flistUpdateInputFile function."""

    def test_updates_multiple_parameters(self):
        """Test updating multiple parameters in a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("# VPlanet input file\n")
            f.write("dTMan 2500\n")
            f.write("dTCMB 4000\n")
            f.write("dAge 4.5e9\n")
            f.name_path = f.name

        try:
            params = [('dTMan', 3000.0), ('dTCMB', 4500.0)]
            result = _flistUpdateInputFile(Path(f.name_path), params)

            result_text = ''.join(result)
            assert '3000' in result_text
            assert '4500' in result_text
            assert '4.5e9' in result_text  # dAge unchanged
            assert '# VPlanet input file' in result_text
        finally:
            os.unlink(f.name_path)

    def test_preserves_file_structure(self):
        """Test that file structure (blank lines, comments) is preserved."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
            f.write("# Header comment\n")
            f.write("\n")
            f.write("dParam1 100\n")
            f.write("\n")
            f.write("# Section 2\n")
            f.write("dParam2 200\n")
            f.name_path = f.name

        try:
            params = [('dParam1', 150.0)]
            result = _flistUpdateInputFile(Path(f.name_path), params)

            assert result[0] == "# Header comment\n"
            assert result[1] == "\n"
            assert result[3] == "\n"
            assert result[4] == "# Section 2\n"
        finally:
            os.unlink(f.name_path)


class TestGenerateMaxLEVInputFiles:
    """Tests for fnGenerateMaxLEVInputFiles function."""

    def test_generates_maxlev_files(self):
        """Test that *_maxlev.in files are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "earth.in"
            input_file.write_text("dTMan 2500\ndTCMB 4000\n")

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(bIsShared=False, sFileName='earth', sParamName='dTMan'),
                MagicMock(bIsShared=False, sFileName='earth', sParamName='dTCMB'),
            ]

            best_params = np.array([3000.0, 4500.0])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 1
            output_file = Path(tmpdir) / "earth_maxlev.in"
            assert output_file.exists()

            content = output_file.read_text()
            assert '3000' in content
            assert '4500' in content

    def test_handles_multiple_files(self):
        """Test generating maxlev files for multiple input files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            star_file = Path(tmpdir) / "star.in"
            star_file.write_text("dLuminosity 1.0\n")

            planet_file = Path(tmpdir) / "planet.in"
            planet_file.write_text("dMass 1.0\n")

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(bIsShared=False, sFileName='star', sParamName='dLuminosity'),
                MagicMock(bIsShared=False, sFileName='planet', sParamName='dMass'),
            ]

            best_params = np.array([0.9, 1.5])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 2
            assert (Path(tmpdir) / "star_maxlev.in").exists()
            assert (Path(tmpdir) / "planet_maxlev.in").exists()

    def test_warns_on_missing_template(self):
        """Test warning when template file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(bIsShared=False, sFileName='nonexistent', sParamName='dParam'),
            ]

            best_params = np.array([100.0])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 0


class MockParam:
    """Mock parameter for testing."""
    def __init__(self, name, bounds, sFileName=None, sParamName=None):
        self.name = name
        self.bounds = bounds
        self.sFileName = sFileName or name.split('.')[0]
        self.sParamName = sParamName or (name.split('.')[1] if '.' in name else name)
        self.bIsShared = False
        self.bodies = None

    def fsSharedTag(self):
        if self.bIsShared:
            return f"  [shared: {', '.join(self.bodies)}]"
        return ""


class MockObservable:
    """Mock observable for testing."""
    def __init__(self, name, observed_value, uncertainty):
        self.name = name
        self.observed_value = observed_value
        self._uncertainty = uncertainty

    def fdGetUncertainty(self, dComputed):
        return self._uncertainty


class TestSaveResults:
    """Tests for fnSaveResults function."""

    def test_saves_results_file(self):
        """Test that results are saved to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_file = Path(tmpdir) / "results.txt"
            input_file = Path(tmpdir) / "test.in"
            input_file.write_text("dParam 100\n")

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='test.dParam', bounds=(0, 200))
            ]
            config.observables = [
                MockObservable(name='Obs1', observed_value=50.0, uncertainty=5.0)
            ]
            config.vplanet = {'inpath': tmpdir}

            model = MagicMock()
            model.fdaRunSimulation.return_value = np.array([52.0])
            model.observable_computer.compute.return_value = {'Obs1': 52.0}

            output_settings = {'results_file': str(results_file)}

            best_params = np.array([150.0])
            best_value = 0.4

            fnSaveResults(best_params, best_value, config, model, output_settings)

            assert results_file.exists()
            content = results_file.read_text()
            assert 'TestRun' in content
            assert 'Maximum Likelihood' in content
            assert '1.500000e+02' in content or '150' in content

    def test_handles_simulation_failure_gracefully(self):
        """Test that fnSaveResults handles simulation failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_file = Path(tmpdir) / "results.txt"

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='test.dParam', bounds=(0, 200))
            ]
            config.observables = []
            config.vplanet = {'inpath': tmpdir}

            model = MagicMock()
            model.fdaRunSimulation.side_effect = Exception("Simulation failed")

            output_settings = {'results_file': str(results_file)}

            best_params = np.array([100.0])
            best_value = 1e10

            fnSaveResults(best_params, best_value, config, model, output_settings)

            assert results_file.exists()
            content = results_file.read_text()
            assert 'Could not compute' in content or 'Maximum Likelihood' in content

    def test_saves_shared_tag_in_results(self):
        """Test that shared param tags appear in the results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_file = Path(tmpdir) / "results.txt"

            param = MockParam(name='dIceAlbedo', bounds=(0.4, 0.8))
            param.bIsShared = True
            param.bodies = ['planet1', 'planet2']

            config = MagicMock()
            config.name = "SharedTest"
            config.parameters = [param]
            config.observables = []
            config.vplanet = {'inpath': tmpdir}

            model = MagicMock()
            model.fdaRunSimulation.return_value = None

            output_settings = {'results_file': str(results_file)}

            fnSaveResults(np.array([0.6]), 1.0, config, model, output_settings)

            content = results_file.read_text()
            assert '[shared: planet1, planet2]' in content


class TestPlotEvolution:
    """Tests for fnPlotEvolution function."""

    def test_skips_when_disabled(self):
        """Test that plotting is skipped when disabled."""
        output_settings = {'plot_evolution': False}
        config = MagicMock()
        model = MagicMock()

        fnPlotEvolution(np.array([1.0]), config, model, output_settings)

        model.fdaRunSimulation.assert_not_called()

    def test_warns_when_no_stop_time(self):
        """Test warning when stop time parameter not found."""
        output_settings = {
            'plot_evolution': True,
            'evolution': {'time_range': [0.1, 13.0], 'num_points': 5}
        }

        config = MagicMock()
        config.parameters = [
            MockParam(name='other.dMass', bounds=(0.5, 1.5))
        ]
        config.observables = []

        model = MagicMock()

        with patch('builtins.print') as mock_print:
            fnPlotEvolution(np.array([1.0]), config, model, output_settings)
            calls = [str(c) for c in mock_print.call_args_list]
            assert any('stop time' in str(c).lower() for c in calls)

    def test_linear_time_scale(self):
        """Test plot evolution with linear time scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_file = Path(tmpdir) / "test_plot.pdf"

            output_settings = {
                'plot_evolution': True,
                'plot_file': str(plot_file),
                'evolution': {
                    'time_range': [0.1, 5.0],
                    'num_points': 3,
                    'log_scale': False
                }
            }

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='earth.dStopTime', bounds=(0.1, 15.0))
            ]
            config.observables = [
                MockObservable(name='HeatFlow', observed_value=47.0, uncertainty=2.0)
            ]

            model = MagicMock()
            model.fdaRunSimulation.return_value = np.array([45.0])
            model.observable_computer.compute.return_value = {'HeatFlow': 45.0}

            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.subplots') as mock_subplots:
                    mock_fig = MagicMock()
                    mock_ax = MagicMock()
                    mock_subplots.return_value = (mock_fig, mock_ax)

                    fnPlotEvolution(np.array([4.5]), config, model, output_settings)

                    mock_ax.set_xscale.assert_not_called()

    def test_successful_plot_generation(self):
        """Test full plot generation with successful simulations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_file = Path(tmpdir) / "test_plot.pdf"

            output_settings = {
                'plot_evolution': True,
                'plot_file': str(plot_file),
                'evolution': {
                    'time_range': [0.1, 5.0],
                    'num_points': 3,
                    'log_scale': True
                }
            }

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='earth.dStopTime', bounds=(0.1, 15.0))
            ]
            config.observables = [
                MockObservable(name='HeatFlow', observed_value=47.0, uncertainty=2.0)
            ]

            model = MagicMock()
            model.fdaRunSimulation.return_value = np.array([45.0])
            model.observable_computer.compute.return_value = {'HeatFlow': 45.0}

            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                fnPlotEvolution(np.array([4.5]), config, model, output_settings)

                mock_ax.set_xscale.assert_called_with('log')
                mock_ax.set_yscale.assert_called_with('log')
                mock_fig.savefig.assert_called_once()

    def test_plot_handles_simulation_failure(self):
        """Test that plot handles simulation failures gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_file = Path(tmpdir) / "test_plot.pdf"

            output_settings = {
                'plot_evolution': True,
                'plot_file': str(plot_file),
                'evolution': {
                    'time_range': [0.1, 2.0],
                    'num_points': 2,
                    'log_scale': True
                }
            }

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='earth.dStopTime', bounds=(0.1, 15.0))
            ]
            config.observables = [
                MockObservable(name='HeatFlow', observed_value=47.0, uncertainty=2.0)
            ]

            model = MagicMock()
            model.fdaRunSimulation.return_value = None  # Simulation fails

            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                fnPlotEvolution(np.array([4.5]), config, model, output_settings)

                mock_fig.savefig.assert_called_once()

    def test_plot_handles_compute_error(self):
        """Test that plot handles observable computation errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_file = Path(tmpdir) / "test_plot.pdf"

            output_settings = {
                'plot_evolution': True,
                'plot_file': str(plot_file),
                'evolution': {
                    'time_range': [0.1, 2.0],
                    'num_points': 2,
                    'log_scale': True
                }
            }

            config = MagicMock()
            config.name = "TestRun"
            config.parameters = [
                MockParam(name='earth.dStopTime', bounds=(0.1, 15.0))
            ]
            config.observables = [
                MockObservable(name='HeatFlow', observed_value=47.0, uncertainty=2.0)
            ]

            model = MagicMock()
            model.fdaRunSimulation.return_value = np.array([45.0])
            model.observable_computer.compute.side_effect = Exception("Compute error")

            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                fnPlotEvolution(np.array([4.5]), config, model, output_settings)

                mock_fig.savefig.assert_called_once()


class TestSharedParameterOutput:
    """Tests for shared parameter output file generation."""

    def test_generates_files_for_all_bodies(self):
        """Test that shared params write to every body file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for sName in ['planet1', 'planet2', 'planet3']:
                pathFile = Path(tmpdir) / f"{sName}.in"
                pathFile.write_text("dIceAlbedo 0.5\n")

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(
                    bIsShared=True,
                    bodies=['planet1', 'planet2', 'planet3'],
                    sParamName='dIceAlbedo',
                    sFileName='planet1',
                ),
            ]

            best_params = np.array([0.65])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 3
            for sName in ['planet1', 'planet2', 'planet3']:
                pathOutput = Path(tmpdir) / f"{sName}_maxlev.in"
                assert pathOutput.exists()
                sContent = pathOutput.read_text()
                assert '0.65' in sContent

    def test_mixed_shared_and_non_shared(self):
        """Test output with both shared and non-shared params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "planet1.in").write_text("dIceAlbedo 0.5\n")
            (Path(tmpdir) / "planet2.in").write_text("dIceAlbedo 0.5\n")
            (Path(tmpdir) / "star.in").write_text("dMass 1.0\n")

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(
                    bIsShared=True,
                    bodies=['planet1', 'planet2'],
                    sParamName='dIceAlbedo',
                    sFileName='planet1',
                ),
                MagicMock(
                    bIsShared=False,
                    bodies=None,
                    sParamName='dMass',
                    sFileName='star',
                ),
            ]

            best_params = np.array([0.65, 0.9])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 3
            assert (Path(tmpdir) / "planet1_maxlev.in").exists()
            assert (Path(tmpdir) / "planet2_maxlev.in").exists()
            assert (Path(tmpdir) / "star_maxlev.in").exists()

    def test_shared_and_nonshared_same_body(self):
        """Test shared+non-shared params writing to the same body file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "planet1.in").write_text(
                "dIceAlbedo 0.5\ndObliquity 23.44\n"
            )
            (Path(tmpdir) / "planet2.in").write_text(
                "dIceAlbedo 0.5\n"
            )

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(
                    bIsShared=True,
                    bodies=['planet1', 'planet2'],
                    sParamName='dIceAlbedo',
                    sFileName='planet1',
                ),
                MagicMock(
                    bIsShared=False,
                    bodies=None,
                    sParamName='dObliquity',
                    sFileName='planet1',
                ),
            ]

            best_params = np.array([0.65, 30.0])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            sContent = (Path(tmpdir) / "planet1_maxlev.in").read_text()
            assert '0.65' in sContent
            assert '30' in sContent
            assert (Path(tmpdir) / "planet2_maxlev.in").exists()

    def test_shared_with_missing_body_file(self):
        """Test shared param when one body file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "planet1.in").write_text("dIceAlbedo 0.5\n")

            config = MagicMock()
            config.vplanet = {'inpath': tmpdir}
            config.parameters = [
                MagicMock(
                    bIsShared=True,
                    bodies=['planet1', 'planet2'],
                    sParamName='dIceAlbedo',
                    sFileName='planet1',
                ),
            ]

            best_params = np.array([0.65])
            result = fnGenerateMaxLEVInputFiles(best_params, config)

            assert len(result) == 1
            assert (Path(tmpdir) / "planet1_maxlev.in").exists()

    def test_flistFileNamesForParameter_shared(self):
        """Test helper returns all bodies for shared param."""
        from maxlev.output import _flistFileNamesForParameter

        param = MagicMock()
        param.bIsShared = True
        param.bodies = ['planet1', 'planet2', 'planet3']

        result = _flistFileNamesForParameter(param)
        assert result == ['planet1', 'planet2', 'planet3']

    def test_flistFileNamesForParameter_non_shared(self):
        """Test helper returns single file for non-shared param."""
        from maxlev.output import _flistFileNamesForParameter

        param = MagicMock()
        param.bIsShared = False
        param.sFileName = 'earth'

        result = _flistFileNamesForParameter(param)
        assert result == ['earth']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
