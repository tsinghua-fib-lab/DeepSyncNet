from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass

from . import differentiation
from . import feature_library
from . import optimizers
from . import utils
from .pysindy import SINDy
from .pysindy import AxesArray
from .differentiation import BaseDifferentiation
from .differentiation import FiniteDifference
from .differentiation import SpectralDerivative
from .differentiation import SINDyDerivative
from .differentiation import SmoothedFiniteDifference
from .feature_library import ConcatLibrary
from .feature_library import TensoredLibrary
from .feature_library import GeneralizedLibrary
from .feature_library import CustomLibrary
from .feature_library import FourierLibrary
from .feature_library import IdentityLibrary
from .feature_library import PolynomialLibrary
from .feature_library import PDELibrary
from .feature_library import WeakPDELibrary
from .feature_library import SINDyPILibrary
from .feature_library import ParameterizedLibrary
from .optimizers import BaseOptimizer
from .optimizers import ConstrainedSR3
from .optimizers import FROLS

try:  # Waiting on PEP 690 to lazy import gurobipy
    from .optimizers import MIOSR
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import CVXPY
    from .optimizers import SINDyPI
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import CVXPY
    from .optimizers import TrappingSR3
except ImportError:
    pass
try:  # Waiting on PEP 690 to lazy import CVXPY
    from .optimizers import StableLinearSR3
except ImportError:
    pass
from .optimizers import WrappedOptimizer
from .optimizers import SR3
from .optimizers import SSR
from .optimizers import STLSQ
from .optimizers import EnsembleOptimizer


__all__ = ["SINDy", "AxesArray"]
__all__.extend(differentiation.__all__)
__all__.extend(feature_library.__all__)
__all__.extend(optimizers.__all__)
__all__.extend(["utils"])
