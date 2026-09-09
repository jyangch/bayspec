import numpy as np

from bayspec.data import Response


def test_channel_fwhm_reproduces_heasp_peak_walk_and_energy_mapping():
    chbin = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ]
    )
    phbin = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    drm = np.array(
        [
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [4.0, 3.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    response = Response(chbin, phbin, drm)

    np.testing.assert_array_equal(response.channel_fwhm, [2.0, 2.0, 4.0, -1.0, -1.0])
