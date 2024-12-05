"""
##############################
# The Kinase Library Package #
##############################
"""

from .objects.core import *
from .objects.phosphoproteomics import *

from .enrichment.binary_enrichment import *
from .enrichment.differential_phosphorylation import *
from .enrichment.mea import *

from .utils import _global_vars
from .utils.utils import *

from .modules.data import *
from .modules.enrichment import *

#%%

__version__ = "1.0.1"

#%% Loading scored phosphoproteome one time per session

_global_vars.all_scored_phosprot = ScoredPhosphoProteome(phosprot_name=_global_vars.phosprot_name, phosprot_path=_global_vars.phosprot_path)