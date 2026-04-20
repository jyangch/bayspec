"""Local spectral components shipped with bayspec.

Re-exports every additive, multiplicative, and mathematical component
defined in the submodules and tracks them in ``local_models`` for
discoverability.
"""

from .additive import *
from .mathematic import *
from .multiplicative import *
from ..model import Model


local_models = {name: cls for name, cls in globals().items()
                if isinstance(cls, type)
                and issubclass(cls, Model)
                and name not in ['Model', 'Additive', 'Multiplicative', 'Mathematic']}


def list_local_models():
    """Return the names of every registered local model class."""

    return list(local_models.keys())

__all__ = list(local_models.keys()) + ['list_local_models', 'local_models']
