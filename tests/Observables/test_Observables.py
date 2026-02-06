"""Test observable computation from VPlanet outputs."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.observables import ObservableComputer
from maxlev.config import ObservableConfig


class TestObservableComputer:
    """Tests for ObservableComputer class."""

    def test_init_sorts_output_names(self):
        """Test that output names are sorted alphabetically."""
        output_names = ['z_output', 'a_output', 'm_output']
        observables = []

        computer = ObservableComputer(output_names, observables)

        assert computer.output_names == ['a_output', 'm_output', 'z_output']

    def test_init_builds_output_index(self):
        """Test that output index maps names to sorted positions."""
        output_names = ['z_output', 'a_output', 'm_output']
        observables = []

        computer = ObservableComputer(output_names, observables)

        assert computer.output_index['a_output'] == 0
        assert computer.output_index['m_output'] == 1
        assert computer.output_index['z_output'] == 2

    def test_compute_direct_observable(self):
        """Test computing direct observables from outputs."""
        output_names = ['final.earth.HeatFlow', 'final.earth.RIC']
        observables = [
            ObservableConfig(
                name='HeatFlow',
                type='direct',
                output='final.earth.HeatFlow',
                observed_value=47.0,
                uncertainty=2.0
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([45.0, 1221.0])
        result = computer.compute(outputs)

        assert 'HeatFlow' in result
        assert result['HeatFlow'] == 45.0

    def test_compute_derived_observable_log10(self):
        """Test computing derived observables with log10 expression."""
        output_names = ['MagMom']
        observables = [
            ObservableConfig(
                name='LogMagMom',
                type='derived',
                expression='log10(MagMom)',
                observed_value=23.0,
                uncertainty=0.5
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([1e22])
        result = computer.compute(outputs)

        assert 'LogMagMom' in result
        assert abs(result['LogMagMom'] - 22.0) < 1e-10

    def test_compute_derived_observable_arithmetic(self):
        """Test computing derived observables with arithmetic expression."""
        output_names = ['a_value', 'b_value']
        observables = [
            ObservableConfig(
                name='Ratio',
                type='derived',
                expression='a_value / b_value',
                observed_value=2.0,
                uncertainty=0.1
            ),
            ObservableConfig(
                name='Sum',
                type='derived',
                expression='a_value + b_value',
                observed_value=15.0,
                uncertainty=1.0
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([10.0, 5.0])
        result = computer.compute(outputs)

        assert result['Ratio'] == 2.0
        assert result['Sum'] == 15.0

    def test_compute_derived_observable_with_sqrt(self):
        """Test derived observable with sqrt function."""
        output_names = ['variance']
        observables = [
            ObservableConfig(
                name='StdDev',
                type='derived',
                expression='sqrt(variance)',
                observed_value=3.0,
                uncertainty=0.1
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([9.0])
        result = computer.compute(outputs)

        assert result['StdDev'] == 3.0

    def test_compute_derived_observable_with_exp(self):
        """Test derived observable with exp function."""
        output_names = ['log_value']
        observables = [
            ObservableConfig(
                name='Value',
                type='derived',
                expression='exp(log_value)',
                observed_value=7.389,
                uncertainty=0.1
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([2.0])
        result = computer.compute(outputs)

        assert abs(result['Value'] - np.exp(2.0)) < 1e-10

    def test_compute_derived_observable_with_abs(self):
        """Test derived observable with abs function."""
        output_names = ['signed_value']
        observables = [
            ObservableConfig(
                name='AbsValue',
                type='derived',
                expression='abs(signed_value)',
                observed_value=5.0,
                uncertainty=0.1
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([-5.0])
        result = computer.compute(outputs)

        assert result['AbsValue'] == 5.0

    def test_compute_mixed_observables(self):
        """Test computing both direct and derived observables."""
        output_names = ['a', 'b', 'c']
        observables = [
            ObservableConfig(
                name='DirectA',
                type='direct',
                output='a',
                observed_value=1.0,
                uncertainty=0.1
            ),
            ObservableConfig(
                name='DerivedBC',
                type='derived',
                expression='b * c',
                observed_value=6.0,
                uncertainty=0.5
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([1.5, 2.0, 3.0])
        result = computer.compute(outputs)

        assert result['DirectA'] == 1.5
        assert result['DerivedBC'] == 6.0

    def test_eval_expression_with_numpy(self):
        """Test that numpy functions are available in expressions."""
        output_names = ['x']
        observables = [
            ObservableConfig(
                name='NumpyExp',
                type='derived',
                expression='np.exp(x)',
                observed_value=2.718,
                uncertainty=0.01
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([1.0])
        result = computer.compute(outputs)

        assert abs(result['NumpyExp'] - np.e) < 1e-10

    def test_eval_expression_invalid_raises(self):
        """Test that invalid expressions raise ValueError."""
        output_names = ['x']
        observables = [
            ObservableConfig(
                name='Invalid',
                type='derived',
                expression='undefined_function(x)',
                observed_value=1.0,
                uncertainty=0.1
            )
        ]

        computer = ObservableComputer(output_names, observables)

        outputs = np.array([1.0])

        with pytest.raises(ValueError, match="Failed to evaluate expression"):
            computer.compute(outputs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
