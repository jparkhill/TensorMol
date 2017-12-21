"""Code Conventions and Style Guide:

- Write code modularly. In proper directories, with minimal imports.
- USE HARD TABS. configure whatever editor you are using to use hard tabs.
- UseCapitalizationToSeparateWords in names.
- Prefer long interperable words to ambiguous abbreviations. MakesDipoleTensor() >> mdt123()
- Avoid_the_underscore to separate words which takes longer to type than a cap. MakesDipoleTensor() >> Makes_Dipole_Tensor
- The underscore is a good way to denote a function argument. TakesSet(aset_)
- Keep functions to fewer than 5 parameters
- Keep files and classes to < 2000 lines.
- Keep classes to < 20 member variables.
- Keep loops to a depth < 6
- Use functional programming constructs whenever possible.
- Use Google-style docstrings, you asshole, and use Args: and Returns:
- Commit your changes once a day at least.
- Use tf.Tensor and np.array rather than python list whenever possible
- It's NOT okay to put default parameters in __init__() and change them all the time. Add them to TMPARAMS.py so they become logged and attached to results.
- import TensorMol as tm; works as desired, don't mess that up.

Violators are subject to having their code and reproductive fitness mocked publically in comments.
"""

from __future__ import absolute_import
from __future__ import print_function
__version__="0.2"
from TensorMol.Util import * # Populates the PARAMS and LOGGER.
from TensorMol.PhysicalData import *
from .Containers import *
from .Math import *
from .TFNetworks import *
from .Interfaces import *
from .ForceModels import *
from .ForceModifiers import *
from .Simulations import *
#__all__ = ["Util", "PhysicalData", "Math" , "Containers" ,"Containers.Mol" ,"Containers.Sets" ,"TFNetworks","Interfaces","ForceModels","ForceModifiers","Simulations"]
LOGGER.debug("TensorMol import complete.")
