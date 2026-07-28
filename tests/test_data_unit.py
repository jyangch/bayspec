import numpy as np

from bayspec.data import DataUnit, Response, Source


def make_response(num_channels):
    chbin = np.column_stack([np.arange(num_channels), np.arange(1, num_channels + 1)])
    phbin = np.array([[0.0, float(num_channels)]])
    drm = np.ones((1, num_channels))

    return Response(chbin, phbin, drm)


def test_notc_boundary_drops_truncated_grouping_slice():
    src = Source(
        counts=np.array([1.0, 2.0, 3.0, 4.0]),
        errors=np.ones(4),
        exposure=1.0,
        grouping=np.array([1, -1, -1, 1]),
    )

    data = DataUnit(src, rsp=make_response(4), notc=[0, 2])

    assert data.grouping_slice.shape == (0, 2)
    assert data.grouping_slice.tolist() == []
    assert data.rebining_slice.shape == (0, 2)
    assert data.rebining_slice.tolist() == []
    np.testing.assert_array_equal(data.src_counts, np.array([]))


def test_notc_boundary_keeps_complete_grouping_slice():
    src = Source(
        counts=np.array([1.0, 2.0, 3.0, 4.0]),
        errors=np.ones(4),
        exposure=1.0,
        grouping=np.array([1, -1, 1, -1]),
    )

    data = DataUnit(src, rsp=make_response(4), notc=[0, 2])

    np.testing.assert_array_equal(data.grouping_slice, np.array([[0, 2]]))
    np.testing.assert_array_equal(data.rebining_slice, np.array([[0, 2]]))
    np.testing.assert_array_equal(data.src_counts, np.array([3.0]))


def test_quality_boundary_drops_truncated_grouping_slice():
    src = Source(
        counts=np.array([1.0, 2.0, 3.0, 4.0]),
        errors=np.ones(4),
        exposure=1.0,
        quality=np.array([0, 0, 1, 0]),
        grouping=np.array([1, -1, -1, 1]),
    )

    data = DataUnit(src, rsp=make_response(4), notc=[0, 4])

    np.testing.assert_array_equal(data.grouping_slice, np.array([[3, 4]]))
    np.testing.assert_array_equal(data.rebining_slice, np.array([[3, 4]]))
    np.testing.assert_array_equal(data.src_counts, np.array([4.0]))
