import os
from pathlib import Path
import shutil
import subprocess

from astropy.io import fits
import numpy as np
import pytest

from bayspec.data import DataUnit

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ('source_name', 'response_name'),
    [
        ('examples/HE/he.src', 'examples/HE/he.rsp'),
        ('examples/ME/me.src', 'examples/ME/me.rsp'),
        ('examples/LE/le.src', 'examples/LE/le.rmf'),
    ],
)
def test_optimal_grouping_matches_ftgrouppha(source_name, response_name, tmp_path):
    executable = shutil.which('ftgrouppha')
    headas = os.environ.get('HEADAS')
    if executable is None or headas is None:
        pytest.skip('HEASoft is not available')

    source = ROOT / source_name
    response = ROOT / response_name
    output = tmp_path / 'grouped.pha'
    pfiles = tmp_path / 'pfiles'
    pfiles.mkdir()

    env = os.environ.copy()
    env['HEADASNOQUERY'] = '1'
    env['PFILES'] = f'{pfiles};{Path(headas) / "syspfiles"}'
    subprocess.run(
        [
            executable,
            f'infile={source}',
            'backfile=none',
            f'outfile={output}',
            'grouptype=opt',
            'groupscale=1',
            'minchannel=-1',
            'maxchannel=-1',
            f'respfile={response}',
            'templatefile=none',
            'rows=-',
            'clobber=yes',
            'mode=ql',
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    with fits.open(output) as hdul:
        expected = np.asarray(hdul['SPECTRUM'].data['GROUPING'])

    unit = DataUnit(src=str(source), rsp=str(response), grpg={'method': 'optimal'})

    np.testing.assert_array_equal(unit.grouping, expected)
