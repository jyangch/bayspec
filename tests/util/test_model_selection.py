from copy import deepcopy
from itertools import permutations
import json
import warnings

import numpy as np
import pytest

from bayspec.util import tools


def make_ic(value, n_params=2, criterion='WAIC'):
    higher = criterion == 'lnZ'
    return {
        'n_params': n_params,
        'n_data_points': 2,
        'criteria': {
            criterion: {
                'value': value,
                'higher_is_better': higher,
                'scale': 'log' if higher else 'deviance',
                'warning': False,
            }
        },
    }


@pytest.mark.parametrize('criterion', ['AIC', 'AICc', 'BIC', 'WAIC', 'LOOIC', 'lnZ'])
def test_selection_uses_native_units_and_strict_global_threshold(criterion):
    values = [-10.0, -8.5, -8.0, -7.0]
    if criterion == 'lnZ':
        values = [10.0, 8.5, 8.0, 7.0]
    bundles = {
        name: make_ic(value, n_params, criterion)
        for name, value, n_params in zip(
            ['best_score', 'simpler', 'boundary', 'outside'], values, [4, 3, 2, 1], strict=True
        )
    }

    assert tools.select_model(bundles, criterion, threshold=2.0) == 'simpler'
    assert tools.select_model(bundles, criterion, threshold=1.0) == 'best_score'
    assert tools.select_model(bundles, criterion, threshold=3.1) == 'outside'


def test_equal_parameter_counts_prefer_score_then_name_in_any_input_order():
    bundles = {'Z': make_ic(10.0), 'A': make_ic(10.0), 'B': make_ic(11.0)}
    original = deepcopy(bundles)

    for names in permutations(bundles):
        assert tools.select_model({name: bundles[name] for name in names}) == 'A'
    assert bundles == original


def test_candidate_with_fewer_parameters_beats_better_score():
    assert tools.select_model({'complex': make_ic(10, 3), 'simple': make_ic(11, 2)}) == 'simple'


def test_single_model_is_selected():
    assert tools.select_model({'only': make_ic(10)}) == 'only'


def test_diagnostic_warning_does_not_exclude_model():
    bundles = {'warned': make_ic(10, 1), 'regular': make_ic(11, 2)}
    bundles['warned']['criteria']['WAIC']['warning'] = True

    with pytest.warns(UserWarning, match=r'warned.*WAIC'):
        assert tools.select_model(bundles) == 'warned'


@pytest.mark.parametrize('value', [None, np.nan, np.inf, -np.inf, 'Infinity', '-Infinity', 'bad'])
def test_invalid_criterion_value_is_not_silently_skipped(value):
    with pytest.raises(ValueError, match=r'invalid.*WAIC'):
        tools.select_model({'valid': make_ic(10), 'invalid': make_ic(value)})


@pytest.mark.parametrize('threshold', [0, -1, np.nan, np.inf, None, 'bad', True])
def test_threshold_must_be_positive_and_finite(threshold):
    with pytest.raises(ValueError, match='threshold'):
        tools.select_model({'model': make_ic(10)}, threshold=threshold)


@pytest.mark.parametrize('n_params', [-1, 1.5, None, True])
def test_parameter_count_must_be_nonnegative_integer(n_params):
    with pytest.raises(ValueError, match='n_params'):
        tools.select_model({'model': make_ic(10, n_params)})


@pytest.mark.parametrize('field', ['n_data_points', 'n_params', 'criteria'])
def test_missing_required_metadata_is_reported(field):
    bundle = make_ic(10)
    del bundle[field]
    with pytest.raises(ValueError, match=field):
        tools.select_model({'model': bundle})


def test_missing_criterion_is_reported():
    with pytest.raises(ValueError, match='lnZ'):
        tools.select_model({'model': make_ic(10)}, criterion='lnZ')


@pytest.mark.parametrize('criterion', ['WAIC', 'BIC', 'lnZ'])
@pytest.mark.parametrize('return_details', [False, True])
@pytest.mark.parametrize('partial_loo', [False, True])
def test_other_criteria_do_not_require_looic(criterion, return_details, partial_loo):
    bundles = {'a': make_ic(10, criterion=criterion), 'b': make_ic(11, criterion=criterion)}
    if criterion == 'WAIC':
        bundles = {'a': predictive_ic([4.0, 6.0]), 'b': predictive_ic([4.0, 7.0])}
    if partial_loo:
        bundles['a']['criteria']['LOOIC'] = {'value': None, 'warning': True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = tools.select_model(bundles, criterion, return_details=return_details)
    expected = 'b' if criterion == 'lnZ' else 'a'
    assert (result['best_model'] if return_details else result) == expected
    assert caught == []


@pytest.mark.parametrize('return_details', [False, True])
def test_explicit_looic_selection_rejects_model_missing_looic(return_details):
    bundles = {'a': predictive_ic([4.0, 6.0], criterion='LOOIC'), 'b': make_ic(11)}
    with pytest.raises(ValueError, match=r'b:.*LOOIC'):
        tools.select_model(bundles, 'LOOIC', return_details=return_details)


@pytest.mark.parametrize('field', ['higher_is_better', 'scale'])
def test_inconsistent_score_conventions_are_rejected(field):
    bundles = {'a': make_ic(10), 'b': make_ic(11)}
    bundles['b']['criteria']['WAIC'][field] = True if field == 'higher_is_better' else 'log'
    with pytest.raises(ValueError, match=field):
        tools.select_model(bundles)


def test_mismatched_data_point_counts_are_rejected():
    bundles = {'a': make_ic(10), 'b': make_ic(11)}
    bundles['b']['n_data_points'] = 3
    with pytest.raises(ValueError, match='n_data_points'):
        tools.select_model(bundles)


@pytest.mark.parametrize('return_details', [False, True])
def test_legacy_data_metadata_is_ignored(return_details):
    bundles = {'a': predictive_ic([4.0, 6.0]), 'b': predictive_ic([4.0, 7.0])}
    expected = tools.select_model(bundles, return_details=return_details)
    bundles['a']['data'] = [{'stat': 'cstat', 'weight': 1}]
    bundles['b']['data'] = [{'stat': 'pgstat', 'weight': 2}]
    assert tools.select_model(bundles, return_details=return_details) == expected


@pytest.mark.parametrize('bundles', [{}, [], {'': make_ic(10)}, {1: make_ic(10)}])
def test_input_requires_named_models(bundles):
    with pytest.raises(ValueError, match='model'):
        tools.select_model(bundles)


def predictive_ic(pointwise, n_params=2, criterion='WAIC'):
    bundle = make_ic(sum(pointwise), n_params, criterion)
    bundle['criteria'][criterion]['pointwise'] = pointwise
    if criterion == 'LOOIC':
        bundle['criteria'][criterion].update(
            pareto_k=[0.2, 0.3],
            good_k=0.7,
            nearly_constant=[False, False],
            psis_failed=[False, False],
        )
    return bundle


@pytest.mark.parametrize('criterion', ['WAIC', 'LOOIC'])
def test_details_use_paired_errors_for_both_references(criterion):
    bundles = {
        'complex': predictive_ic([2.0, 8.0], 3, criterion),
        'simple': predictive_ic([4.0, 7.0], 2, criterion),
        'third': predictive_ic([6.0, 6.0], 4, criterion),
    }
    original = deepcopy(bundles)
    result = tools.select_model(bundles, criterion, return_details=True)

    assert result['best_model'] == 'simple'
    assert result['highest_score_model'] == 'complex'
    assert result['candidate_models'] == ['complex', 'simple']
    selected = result['comparisons']['selected']
    highest = result['comparisons']['highest_score']
    assert selected['reference_model'] == 'simple'
    assert highest['reference_model'] == 'complex'
    assert highest['models']['simple']['delta'] == 1.0
    assert selected['models']['complex']['delta'] == -1.0
    # Pointwise differences [2, -1]: N * population variance = 4.5.
    assert highest['models']['simple']['delta_error'] == pytest.approx(np.sqrt(4.5))
    assert selected['models']['complex']['delta_error'] == pytest.approx(np.sqrt(4.5))
    assert highest['models']['complex']['delta_error'] == 0.0
    assert selected['models']['simple']['delta_error'] == 0.0
    assert selected['models']['third']['delta'] == 1.0
    assert highest['models']['third']['delta'] == 2.0
    assert selected['models']['third']['delta_error'] == pytest.approx(np.sqrt(4.5))
    assert highest['models']['third']['delta_error'] == pytest.approx(np.sqrt(18.0))
    assert result['models']['simple']['diagnostics']['status'] == 'no_warning'
    assert selected['models']['complex']['comparison_status'] == 'no_warning'
    assert tools.select_model(bundles, criterion) == result['best_model']
    assert bundles == original
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert (
        tools.select_model(dict(reversed(list(bundles.items()))), criterion, return_details=True)
        == result
    )


def test_constant_pointwise_difference_has_zero_paired_error():
    bundles = {'A': predictive_ic([1.0, 100.0]), 'B': predictive_ic([2.0, 101.0])}
    result = tools.select_model(bundles, return_details=True)
    highest = result['comparisons']['highest_score']
    assert highest['models']['B']['delta'] == 2.0
    assert highest['models']['B']['delta_error'] == 0.0
    assert result['comparisons']['selected'] == highest
    assert result['candidate_models'] == ['A']


@pytest.mark.parametrize(
    'flag, status', [(False, 'no_warning'), (True, 'warning'), (None, 'insufficient')]
)
def test_waic_diagnostic_status(flag, status):
    bundle = predictive_ic([1.0, 2.0])
    bundle['criteria']['WAIC']['warning'] = flag
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = tools.select_model({'model': bundle}, return_details=True)
    diagnostic = result['models']['model']['diagnostics']
    assert diagnostic['status'] == status
    if status != 'no_warning':
        assert diagnostic['reasons']


@pytest.mark.parametrize(
    'k, constant, failed, status',
    [
        (None, True, False, 'no_warning'),
        (None, False, True, 'warning'),
        (None, False, False, 'insufficient'),
        ('Infinity', False, False, 'warning'),
        (0.8, False, False, 'warning'),
    ],
)
def test_loo_diagnostics_distinguish_undefined_tail_provenance(k, constant, failed, status):
    bundle = predictive_ic([1.0, 2.0], criterion='LOOIC')
    bundle['criteria']['LOOIC'].update(
        pareto_k=[k, 0.2],
        nearly_constant=[constant, False],
        psis_failed=[failed, False],
    )
    result = tools.select_model({'model': bundle}, 'LOOIC', return_details=True)
    diagnostic = result['models']['model']['diagnostics']
    assert diagnostic['status'] == status
    assert diagnostic['unexplained_k_channels'] == (
        [0] if k is None and not constant and not failed else []
    )


def test_legacy_null_pareto_k_has_insufficient_diagnostics():
    bundle = predictive_ic([1.0, 2.0], criterion='LOOIC')
    result = bundle['criteria']['LOOIC']
    result['pareto_k'][0] = None
    del result['nearly_constant']
    del result['psis_failed']
    details = tools.select_model({'model': bundle}, 'LOOIC', return_details=True)
    assert details['models']['model']['diagnostics']['status'] == 'insufficient'


def test_reference_warning_is_scoped_to_each_comparison_group():
    bundles = {
        'complex': predictive_ic([2.0, 8.0], 3),
        'simple': predictive_ic([4.0, 7.0], 2),
        'third': predictive_ic([6.0, 6.0], 4),
    }
    bundles['complex']['criteria']['WAIC']['warning'] = True
    with pytest.warns(UserWarning):
        result = tools.select_model(bundles, return_details=True)
    assert result['best_model'] == 'simple'
    assert result['models']['simple']['diagnostics']['status'] == 'no_warning'
    highest = result['comparisons']['highest_score']['models']
    selected = result['comparisons']['selected']['models']
    assert highest['simple']['comparison_status'] == 'warning'
    assert highest['third']['comparison_status'] == 'warning'
    assert selected['third']['comparison_status'] == 'no_warning'
    assert selected['complex']['comparison_status'] == 'warning'
    assert highest['simple']['delta_error'] == pytest.approx(np.sqrt(4.5))


@pytest.mark.parametrize(
    'pointwise', [None, [1.0], [[1.0, 2.0]], [1.0, None], [1.0, np.inf], [1.0, 9.0]]
)
def test_invalid_pointwise_data_cannot_produce_comparison_errors(pointwise):
    bundle = predictive_ic([1.0, 2.0])
    bundle['criteria']['WAIC']['pointwise'] = pointwise
    with pytest.raises(ValueError, match='pointwise'):
        tools.select_model({'model': bundle}, return_details=True)


def test_one_channel_does_not_claim_zero_uncertainty():
    bundle = predictive_ic([3.0])
    bundle['n_data_points'] = 1
    result = tools.select_model({'model': bundle}, return_details=True)
    for comparison in result['comparisons'].values():
        assert comparison['models']['model']['delta_error'] is None
        assert comparison['models']['model']['comparison_status'] == 'insufficient'


@pytest.mark.parametrize('criterion', ['BIC', 'lnZ'])
def test_nonpredictive_details_do_not_invent_uncertainties(criterion):
    simple_value = 9 if criterion == 'lnZ' else 11
    result = tools.select_model(
        {'complex': make_ic(10, 3, criterion), 'simple': make_ic(simple_value, 2, criterion)},
        criterion,
        return_details=True,
    )
    assert result['best_model'] == 'simple'
    assert result['highest_score_model'] == 'complex'
    assert result['comparisons']['selected']['models']['complex']['delta'] == -1.0
    assert result['comparisons']['highest_score']['models']['simple']['delta'] == 1.0
    for comparison in result['comparisons'].values():
        assert comparison['models']['simple']['delta_error'] is None
        assert comparison['models']['simple']['comparison_status'] == 'not_assessed'
    assert result['models']['simple']['diagnostics']['status'] == 'not_assessed'


def test_equal_scores_keep_name_tiebreak_for_highest_and_parameter_tiebreak_for_selected():
    bundles = {'A': predictive_ic([1.0, 9.0], 3), 'Z': predictive_ic([2.0, 8.0], 2)}
    result = tools.select_model(bundles, return_details=True)
    assert result['best_model'] == 'Z'
    assert result['highest_score_model'] == 'A'
    assert result['comparisons']['selected']['models']['A']['delta'] == 0.0
    assert result['comparisons']['selected']['models']['A']['delta_error'] == pytest.approx(
        np.sqrt(2)
    )
    assert result['comparisons']['highest_score']['models']['Z']['delta_error'] == pytest.approx(
        np.sqrt(2)
    )
