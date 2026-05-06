"""bayspec -- Bayesian spectral analysis for high-energy astrophysics.

Re-exports the four functional layers -- ``util`` (priors, parameters,
posteriors, plotting), ``data`` (spectra, responses, data containers),
``model`` (spectral model algebra), and ``infer`` (pairs, fits, analyzers) --
so that downstream code can ``from bayspec import ...`` directly.
"""

from .util import *  # noqa: F403
from .data import *  # noqa: F403
from .model import *  # noqa: F403
from .infer import *  # noqa: F403
from .__info__ import __version__
