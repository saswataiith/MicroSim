"""Validate and arrange TC phase data without requiring a Thermo-Calc licence."""
from pathlib import Path
import numpy as np
import pandas as pd


def output_directory(material, parent=Path('.')):
    if not material or Path(material).name != material or material in ('.', '..'):
        raise ValueError('Material must be a single nonempty directory name')
    target = Path(parent) / ('tdbs_encrypted_' + material)
    target.mkdir(parents=True, exist_ok=True)
    return target


def phase_frame(groups, phase, solutes):
    """Join components by actual temperature rather than their row positions."""
    wanted = {name.upper(): name for name in solutes}
    columns = {}
    for group in groups.values():
        element = group.label.split(' in ', 1)[0].strip().upper()
        if element not in wanted:
            continue
        name = wanted[element]
        frame = pd.DataFrame({'Temperature': group.x, name + '@' + phase: group.y})
        if frame.Temperature.duplicated().any():
            raise ValueError('Duplicate temperature entries for ' + group.label)
        if name in columns:
            raise ValueError('Ambiguous composition-set label for ' + group.label)
        columns[name] = frame
    if set(columns) != set(solutes):
        raise ValueError('Missing phase components for ' + phase)
    frame = columns[solutes[0]]
    for name in solutes[1:]:
        frame = frame.merge(columns[name], on='Temperature', how='inner', validate='one_to_one')
    if (frame.drop(columns='Temperature').sum(axis=1) > 1.0 + 1e-10).any():
        raise ValueError('Independent phase mole fractions exceed unity for ' + phase)
    return frame


def common_phase_frame(frames):
    if not frames:
        raise ValueError('At least one phase table is required')
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on='Temperature', how='inner', validate='one_to_one')
    result = result.sort_values('Temperature').reset_index(drop=True)
    if len(result) < 3:
        raise ValueError('Need at least three common phase-coexistence temperatures for cubic splines')
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite phase data cannot be exported')
    values = result.drop(columns='Temperature')
    if ((values < 0) | (values > 1)).any().any():
        raise ValueError('Phase mole fractions must lie in [0, 1]')
    return result


def write_composition_tables(frame, solid_phases, matrix_phase, solutes, directory=Path('.')):
    for phase in solid_phases:
        columns = ['Temperature'] + [s + '@' + phase for s in solutes]
        columns += [s + '@' + matrix_phase for s in solutes]
        frame[columns].to_csv(Path(directory) / ('Composition_' + phase + '.csv'), index=False)
