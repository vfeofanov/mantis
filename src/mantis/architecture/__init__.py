"""Init file for architecture.""" 

from .version1 import Mantis8M
from .version2 import MantisV2

MantisV1 = Mantis8M

__all__ = ["Mantis8M", "MantisV1", "MantisV2"]
