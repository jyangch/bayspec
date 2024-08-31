"""
BSD 3-Clause License

Copyright (c) 2018, Giacomo Vianello
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
NOTE: this script is basically from gv_significance 
(https://github.com/giacomov/gv_significance.git)
Giacomo Vianello 2018 ApJS: https://doi.org/10.3847/1538-4365/aab780
"""

import numpy as np
from math import log
import scipy.optimize
from numpy import sqrt, squeeze


def xlogy(x, y):
    """
    This function implements x * log(y) so that if both x and y are 0, 
    the results is zero (and not infinite or nan
    as the computer would return otherwise).
    NOTE: x and y must be numbers, not arrays
    """

    if x == 0.0:
        
        return 0.0
    
    else:
        
        return x * log(y)


def xlogyv(x, y):
    """
    This function implements x * log(y) so that if both x and y are 0, 
    the results is zero (and not infinite or nan
    as the computer would return otherwise).
    Version which accepts numpy.array as inputs.
    """

    x = np.array(x, ndmin=1)
    y = np.array(y, ndmin=1)

    results = np.zeros_like(y)

    idx = (x != 0)

    results[idx] = x[idx] * np.log(y[idx])

    return np.squeeze(results)


def size_one_or_n(value, other_array, name):

    value_ = np.array(value, dtype=float, ndmin=1)

    if value_.shape[0] == 1:

        value_ = np.zeros(other_array.shape[0], dtype=float) + value

    else:

        assert value_.shape[0] == other_array.shape[0], \
            f"The size of {name} must be either 1 or the same size of n"

    return value_


def pgsig(n, b, sigma):
    """
    Returns the significance for observing n counts when b are expected. 
    The measurement "b +/- sigma" is returned by some kind of background estimation procedure, 
    and b is assumed to be a Gaussian random variable.
    :param n: observed counts
    :param b: estimation of the background coming from some method
    :param sigma: error on the estimation of the background
    :return: the significance of the measurement(s)
    """

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    sigma_ = size_one_or_n(sigma, n_, "sigma")

    # Assign sign depending on whether n_ > b_

    sign = np.where(n_ >= b_, 1, -1)

    B0_mle = 0.5 * (b_ - sigma_ ** 2 + sqrt(b_ ** 2 - 2 * b_ * sigma_ ** 2 + 4 * n_ * sigma_ ** 2 + sigma_ ** 4))

    # B0_mle could be slightly < 0 (even though it shouldn't) because of the
    # limited numerical precision of the calculator. let's accept as negative as 0.01, and clip
    # at zero to avoid giving results difficult to interpret
    
    assert np.all(B0_mle > -0.01), "This is a bug. B0_mle cannot be negative."

    B0_mle = np.clip(B0_mle, 0, None)

    return squeeze(sqrt(2) * sqrt(xlogyv(n_, n_ / B0_mle) + (b_ - B0_mle)**2 / (2 * sigma_**2) + B0_mle - n_) * sign)


def _li_and_ma(n_, b_, alpha):

    # In order to avoid numerical problem with 0 * log(0), 
    # we add a tiny number to n_ and b_, inconsequential for the computation
    
    n_ += 1E-25  # type: np.ndarray
    b_ += 1E-25  # type: np.ndarray

    # Pre-compute for speed
    n_plus_b = n_ + b_
    ap1 = alpha + 1

    res = n_ * np.log(ap1 / alpha * (n_ / n_plus_b))

    res += b_ * np.log(ap1 * (b_ / n_plus_b))

    return np.sqrt(2 * res)


def _likelihood_with_sys(o, b, a, s, k, B, M):
    # Keep away from unphysical solutions during maximization
    # (where the argument of the logarithm is negative) by returning
    # a large negative number

    if M + a * B <= 0 or k + 1 <= 0 or B <= 0:
        return -1000

    # Pre-compute for speed

    Ba = B * a
    Bak = B * a * k

    res = -Bak - Ba - B - M + xlogyv(b, B) - k ** 2 / (2 * s ** 2) + xlogyv(o, Bak + Ba + M)

    return res


def _get_TS_by_numerical_optimization(n_, b_, alpha, sigma):

    # Numerical optimization to find the maximum likelihood value
    # for the null hypothesis (M=0). We use the closed form for B_mle provided in the paper
    # so that the optimization is in one variable (kk)
    # NOTE: optimize.minimize minimizes, so we multiply the log-likelihood by -1

    wrapper = lambda kk: -1 * _likelihood_with_sys(n_, b_, alpha, sigma, kk,
                                                   B=(b_ + n_) / (alpha * kk + alpha + 1),
                                                   M=0)

    res = scipy.optimize.minimize(wrapper,
                                  [0.0],
                                  tol=1e-3)

    # Get the minimum of the -log likelihood
    h0_mlike_value = res['fun']

    # Get alternative hypothesis minimum of the -log likelihood

    h1_mlike_value = -(xlogy(b_, b_) - b_ + xlogy(n_, n_) - n_)

    # Compute Test Statistic of Likelihood Ratio Test

    TS = 2 * (h0_mlike_value - h1_mlike_value)

    return TS


_get_TS_by_numerical_optimization_v = np.vectorize(_get_TS_by_numerical_optimization)


def ppsig(n, b, alpha, sigma=0, k=0):
    """
    Returns the significance for detecting n counts when alpha * B are expected.

    If sigma=0 and k=0 (default), this is the case with no additional systematic error and the classic result
    from Li & Ma (1983) is used. Example:

        > significance(n, b, alpha)

    If k>0 then eq.7 from Vianello (2018) is used, which assumes that k is the upper boundary on the fractional
    systematic uncertainty. In this case sigma has no meaning and is ignored. Example:

        > significance(n, b, alpha, k=0.1)

    If sigma>0, then eq. 9 from Vianello (2018) is used, which assumes a Gaussian distribution for the systematic
    uncertainty. In this case k has no meaning and is ignored.Example:

        > significance(n, b, alpha, sigma=0.1)

    :param n: observed counts (can be an array)
    :param b: expected background counts (can be an array)
    :param alpha: ratio of the source observation efficiency and background observation efficiency
    (either a float, or an array of the same shape of n)
    :param sigma: standard deviation for the Gaussian case (either a float, or an array of the same shape of n)
    :param k: maximum fractional systematic uncertainty expected (either a float, or an array of the same shape of n)
    :return: the significance (z score) for the measurement(s)
    """

    # Make sure we are dealing with arrays, and if not, make the input so. This way
    # we can unify the treatment

    n_ = np.array(n, dtype=float, ndmin=1)
    b_ = np.array(b, dtype=float, ndmin=1)

    k_ = size_one_or_n(k, n_, "k")

    sigma_ = size_one_or_n(sigma, n_, "sigma")

    alpha_ = size_one_or_n(alpha, n_, "alpha")

    # Assign sign depending on whether n_ > b_

    sign = np.where(n_ >= alpha_ * b_, 1, -1)

    # Prepare vector for results
    res = np.zeros(n_.shape[0], dtype=float)

    # Select elements where we need to apply Li & Ma and apply it
    idx_lima = (sigma_ == 0) & (k_ == 0)

    res[idx_lima] = _li_and_ma(n_[idx_lima], b_[idx_lima], alpha_[idx_lima])

    # Select elements where we need to apply eq. 7 from Vianello (2018),
    # which is simply Li & Ma with alpha -> alpha * (k+1)

    idx_eq7 = (k_ > 0)
    res[idx_eq7] = _li_and_ma(n_[idx_eq7], b_[idx_eq7], alpha_[idx_eq7] * (k_[idx_eq7] + 1))

    # Select elements where we need to apply eq. 9 from Vianello (2018)
    idx_eq9 = (sigma_ > 0)

    if np.any(idx_eq9):

        TS = _get_TS_by_numerical_optimization_v(n_[idx_eq9], b_[idx_eq9], alpha_[idx_eq9], sigma_[idx_eq9])

        res[idx_eq9] = np.sqrt(TS)

    # Return significance

    return np.squeeze(sign * res)
