"""
VPlanet option validation module.

Validates user configuration against VPlanet's valid options by parsing
the output of `vplanet -h`.
"""

import subprocess
import re
from typing import Dict, Set, Tuple, List, Optional
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_vplanet_help(executable: str = "vplanet") -> str:
    """
    Call vplanet -h and return help text.

    Cached to avoid repeated subprocess calls.

    Args:
        executable: Path to vplanet executable

    Returns:
        Help text string

    Raises:
        RuntimeError: If vplanet -h fails
    """
    try:
        result = subprocess.run(
            [executable, "-h"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run '{executable} -h': {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout running '{executable} -h'")
    except FileNotFoundError:
        raise RuntimeError(f"VPlanet executable not found: {executable}")


def parse_vplanet_options(help_text: str) -> Dict[str, Set[str]]:
    """
    Parse vplanet help output to extract valid options.

    Args:
        help_text: Output from `vplanet -h`

    Returns:
        Dictionary with keys:
            - 'input_options': Set of valid input parameter names
            - 'output_params': Set of valid output parameter names
    """
    result = {
        'input_options': set(),
        'output_params': set()
    }

    # Extract input options from "Input Options" section
    # Format: bOptionName (Bool) -- Description ...
    #         dOptionName (Double) -- Description ...
    #         sOptionName (String) -- Description ...
    in_input_section = False
    in_output_section = False

    for line in help_text.split('\n'):
        # Detect sections
        if 'Input Options' in line:
            in_input_section = True
            in_output_section = False
            continue
        elif 'These options follow the argument saOutputOrder' in line:
            in_input_section = False
            in_output_section = True
            continue
        elif line.strip().startswith('===') or line.strip().startswith('---'):
            continue

        # Parse input options
        if in_input_section and line.strip():
            # Match pattern: optionName (Type) -- or [-]optionName (Type) --
            # Remove leading [-] if present
            line_clean = line.lstrip('[-] \t')
            match = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)\s+\(', line_clean)
            if match:
                option_name = match.group(1)
                result['input_options'].add(option_name)

        # Parse output parameters
        if in_output_section and line.strip():
            # Output params can have format:
            # [-]ParamName -- Description. [Units]
            # or just:
            # ParamName -- Description.
            if '--' in line:
                # Extract parameter name before --
                param_part = line.split('--')[0].strip()
                # Remove leading [-] if present
                param_part = param_part.lstrip('[-]').strip()
                # Extract first word as parameter name
                if param_part:
                    param_name = param_part.split()[0] if ' ' in param_part else param_part
                    if param_name and param_name[0].isalpha():
                        result['output_params'].add(param_name)

    return result


def fuzzy_match(name: str, valid_options: Set[str], max_suggestions: int = 5) -> List[str]:
    """
    Find similar option names using fuzzy matching.

    Args:
        name: Invalid option name
        valid_options: Set of valid option names
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of similar option names
    """
    from difflib import get_close_matches
    return get_close_matches(name, valid_options, n=max_suggestions, cutoff=0.6)


def validate_parameter(param_full_name: str, valid_options: Set[str]) -> Tuple[bool, str, List[str]]:
    """
    Check if a parameter name is valid.

    Args:
        param_full_name: Full parameter path (e.g., "star.dMass", "vpl.dStopTime")
        valid_options: Set of valid input option names

    Returns:
        Tuple of (is_valid, error_message, suggestions)
    """
    # Extract parameter name from full path
    # "star.dMass" → "dMass"
    # "vpl.dStopTime" → "dStopTime"
    parts = param_full_name.split('.')
    if len(parts) < 2:
        return False, f"Invalid parameter format '{param_full_name}' (expected 'file.parameter')", []

    param_name = parts[-1]  # Last part is the parameter name

    # Check if parameter is valid
    if param_name in valid_options:
        return True, "", []

    # Parameter is invalid - find suggestions
    suggestions = fuzzy_match(param_name, valid_options)
    error_msg = f"Invalid VPlanet parameter '{param_name}'"

    return False, error_msg, suggestions


def validate_output(output_full_name: str, valid_outputs: Set[str]) -> Tuple[bool, str, List[str]]:
    """
    Check if an output parameter is valid.

    Args:
        output_full_name: Full output path (e.g., "final.star.LXUVStellar")
        valid_outputs: Set of valid output parameter names

    Returns:
        Tuple of (is_valid, error_message, suggestions)
    """
    # Extract output name from full path
    # "final.star.LXUVStellar" → "LXUVStellar"
    parts = output_full_name.split('.')
    if len(parts) < 3:
        return False, f"Invalid output format '{output_full_name}' (expected 'final.body.parameter')", []

    output_name = parts[-1]  # Last part is the output parameter name

    # Check if output is valid
    if output_name in valid_outputs:
        return True, "", []

    # Output is invalid - find suggestions
    suggestions = fuzzy_match(output_name, valid_outputs)
    error_msg = f"Invalid VPlanet output parameter '{output_name}'"

    return False, error_msg, suggestions


def validate_body_name(output_full_name: str, vplanet_inpath: str, vplfile: str = "vpl.in") -> Tuple[bool, str]:
    """
    Check if body name exists in VPlanet input files.

    Args:
        output_full_name: Full output path (e.g., "final.star.LXUVStellar")
        vplanet_inpath: Path to VPlanet input directory
        vplfile: Name of main VPlanet input file

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Extract body name
    # "final.star.LXUVStellar" → "star"
    parts = output_full_name.split('.')
    if len(parts) < 3:
        return False, f"Invalid output format '{output_full_name}' (expected 'final.body.parameter')"

    body_name = parts[1]  # Middle part is the body name

    # Read vpl.in to get saBodyFiles
    vpl_path = Path(vplanet_inpath) / vplfile
    if not vpl_path.exists():
        return False, f"VPlanet input file not found: {vpl_path}"

    try:
        with open(vpl_path, 'r') as f:
            vpl_content = f.read()
    except Exception as e:
        return False, f"Failed to read {vpl_path}: {e}"

    # Parse saBodyFiles to get list of body files
    # Format: saBodyFiles    star.in planet.in
    # Can span multiple lines with $ continuation
    body_files = []
    for line in vpl_content.split('\n'):
        # Remove comments
        line = line.split('#')[0].strip()
        if not line:
            continue

        if line.startswith('saBodyFiles'):
            # Extract files after saBodyFiles
            parts = line.split()
            if len(parts) > 1:
                body_files.extend(parts[1:])
                # TODO: Handle $ continuation if needed

    # Extract body names from file names
    # "star.in" → "star"
    valid_bodies = set()
    for body_file in body_files:
        if body_file.endswith('.in'):
            valid_bodies.add(body_file[:-3])

    if body_name in valid_bodies:
        return True, ""

    error_msg = f"Body '{body_name}' not found in {vplfile}. Valid bodies: {', '.join(sorted(valid_bodies))}"
    return False, error_msg


def validate_saOutputOrder(output_full_name: str, vplanet_inpath: str) -> Tuple[bool, str]:
    """
    Verify output parameter is in body's saOutputOrder.

    This is CRITICAL - ensures VPlanet will actually print this output.

    Args:
        output_full_name: Full output path (e.g., "final.star.LXUVStellar")
        vplanet_inpath: Path to VPlanet input directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Extract body name and output parameter
    parts = output_full_name.split('.')
    if len(parts) < 3:
        return False, f"Invalid output format '{output_full_name}'"

    body_name = parts[1]
    output_param = parts[-1]

    # Read body .in file
    body_file = Path(vplanet_inpath) / f"{body_name}.in"
    if not body_file.exists():
        return False, f"Body file not found: {body_file}"

    try:
        with open(body_file, 'r') as f:
            body_content = f.read()
    except Exception as e:
        return False, f"Failed to read {body_file}: {e}"

    # Parse saOutputOrder
    # Format: saOutputOrder Time -Luminosity -LXUVStellar
    # Can span multiple lines with $ continuation
    output_order = []
    in_output_order = False

    for line in body_content.split('\n'):
        # Remove comments
        line_stripped = line.split('#')[0].strip()
        if not line_stripped:
            continue

        if line_stripped.startswith('saOutputOrder'):
            in_output_order = True
            # Extract parameters after saOutputOrder
            parts = line_stripped.split()
            if len(parts) > 1:
                output_order.extend(parts[1:])
            # Check for continuation
            if not line_stripped.endswith('$'):
                break
        elif in_output_order:
            # Continuation line
            parts = line_stripped.split()
            output_order.extend([p for p in parts if p != '$'])
            if not line_stripped.endswith('$'):
                break

    # Remove leading - from output parameters
    output_order_clean = [p.lstrip('-') for p in output_order]

    # Check if requested output is in the list
    # Note: Output parameters in saOutputOrder are case-insensitive and can be abbreviated
    # For simplicity, we'll do case-insensitive exact match
    output_param_lower = output_param.lower()
    for param in output_order_clean:
        if param.lower() == output_param_lower:
            return True, ""

    error_msg = (
        f"Output '{output_param}' not in saOutputOrder for body '{body_name}'. "
        f"Current saOutputOrder: {', '.join(output_order_clean)}"
    )
    return False, error_msg


def validate_all(config_dict: dict, vplanet_executable: str = "vplanet") -> List[Tuple[str, int]]:
    """
    Validate all parameters and outputs in config.

    Args:
        config_dict: Configuration dictionary
        vplanet_executable: Path to vplanet executable

    Returns:
        List of (error_message, line_number) tuples. Empty if all valid.
    """
    errors = []

    # Get valid options from vplanet -h
    try:
        help_text = get_vplanet_help(vplanet_executable)
        options = parse_vplanet_options(help_text)
    except RuntimeError as e:
        errors.append((f"Failed to get VPlanet help: {e}", 0))
        return errors

    vplanet_inpath = config_dict.get('vplanet', {}).get('inpath', '.')

    # Validate parameters
    for i, param in enumerate(config_dict.get('parameters', [])):
        param_name = param.get('name', '')
        # Estimate line number (rough approximation)
        line_num = 10 + i * 6  # Rough estimate

        is_valid, error_msg, suggestions = validate_parameter(param_name, options['input_options'])
        if not is_valid:
            msg = f"{error_msg}"
            if suggestions:
                msg += f". Did you mean: {', '.join(suggestions)}?"
            errors.append((msg, line_num))

    # Validate outputs
    for i, output in enumerate(config_dict.get('outputs', [])):
        output_name = output.get('name', '')
        line_num = 50 + i * 4  # Rough estimate

        # Validate output parameter name
        is_valid, error_msg, suggestions = validate_output(output_name, options['output_params'])
        if not is_valid:
            msg = f"{error_msg}"
            if suggestions:
                msg += f". Did you mean: {', '.join(suggestions)}?"
            errors.append((msg, line_num))
            continue  # Skip further validation if output name is invalid

        # Validate body name
        is_valid, error_msg = validate_body_name(output_name, vplanet_inpath)
        if not is_valid:
            errors.append((error_msg, line_num))
            continue

        # Validate saOutputOrder
        is_valid, error_msg = validate_saOutputOrder(output_name, vplanet_inpath)
        if not is_valid:
            errors.append((error_msg, line_num))

    # Validate observables (they reference outputs, which we already validated)
    # Just need to check expressions reference valid outputs
    for i, obs in enumerate(config_dict.get('observables', [])):
        line_num = 80 + i * 8  # Rough estimate

        if obs.get('type') == 'direct':
            output_ref = obs.get('output', '')
            # Validate output parameter name
            is_valid, error_msg, suggestions = validate_output(output_ref, options['output_params'])
            if not is_valid:
                msg = f"Observable '{obs.get('name')}': {error_msg}"
                if suggestions:
                    msg += f". Did you mean: {', '.join(suggestions)}?"
                errors.append((msg, line_num))

        elif obs.get('type') == 'derived':
            # Parse expression to find output references
            expression = obs.get('expression', '')
            # Find all references like "final.body.Parameter"
            output_refs = re.findall(r'final\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+', expression)
            for output_ref in output_refs:
                is_valid, error_msg, suggestions = validate_output(output_ref, options['output_params'])
                if not is_valid:
                    msg = f"Observable '{obs.get('name')}' expression: {error_msg}"
                    if suggestions:
                        msg += f". Did you mean: {', '.join(suggestions)}?"
                    errors.append((msg, line_num))
                    continue

                # Validate body name
                is_valid, error_msg = validate_body_name(output_ref, vplanet_inpath)
                if not is_valid:
                    errors.append((f"Observable '{obs.get('name')}' expression: {error_msg}", line_num))
                    continue

                # Validate saOutputOrder
                is_valid, error_msg = validate_saOutputOrder(output_ref, vplanet_inpath)
                if not is_valid:
                    errors.append((f"Observable '{obs.get('name')}' expression: {error_msg}", line_num))

    return errors
