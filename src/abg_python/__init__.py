## unify namespaces for each of the sub-files here for convenience's sake
from .system_utils import *
from .function_utils import *
from .selection_utils import *
from .array_utils import *
from .smooth_utils import *
from .math_utils import *
from .physics_utils import *
#from .cosmo_utils import *
from .fitting_utils import *
#from .snapshot_utils import *
#from .plot_utils import * ## need to answer test, can at least smoke test
from .color_utils import * ## does not make sense to unit test

## define some constants
# Code mass -> g , (code length)^-3 -> cm^-3 , g -> nH
DENSITYFACT = 2e43*(3.086e21)**-3/(1.67e-24)
HYDROGENMASS = 1.67e-24  # g
cm_per_kpc = 3.08e21 # cm/kpc
Gcgs = 6.674e-8 #cm3/(g s^2)
SOLAR_MASS_g = 1.989e33 ## g
