import pytest

from bayspec.util.group import _grouping_significance


@pytest.mark.parametrize('stat', ['Xppstat', 'Xcstat', 'Xpgstat'])
def test_grouping_significance_rejects_x_prefixed_statistics(stat):
    with pytest.raises(AttributeError, match=rf'unsupported stat: {stat}'):
        _grouping_significance(
            src=10.0,
            bkg=2.0,
            bkg_err=1.0,
            alpha=1.0,
            stat=stat,
        )
