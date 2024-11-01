from .util import *
from .data import *
from .model import *
from .infer import *
from .__info__ import __version__


from os.path import dirname, abspath
__app__ = dirname(dirname(abspath(__file__))) + '/app/app.py'