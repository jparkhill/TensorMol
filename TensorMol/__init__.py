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

from TensorMol.Math.Statistics import *
from TensorMol.Math.LinearOperations import *
from TensorMol.Math.Ipecac import *
from TensorMol.Math.EmbOpt import *
from TensorMol.Math.Basis import *
from TensorMol.Math.DIIS import *
from TensorMol.Math.BFGS import *

from TensorMol.Containers.Mol import *
from TensorMol.Containers.Sets import *
from TensorMol.Containers.MolFrag import *
from TensorMol.Containers.Digest import *
from TensorMol.Containers.DigestMol import *
from TensorMol.Containers.TensorData import *
from TensorMol.Containers.TensorMolData import *
from TensorMol.Containers.TensorMolDataEE import *

from TensorMol.TFNetworks.TFInstance import *
from TensorMol.TFNetworks.TFMolInstance import *
from TensorMol.TFNetworks.TFMolInstanceDirect import *
from TensorMol.TFNetworks.TFForces import *
from TensorMol.TFNetworks.TFManage import *
from TensorMol.TFNetworks.TFMolManage import *

from TensorMol.Interfaces.AbInitio import *

from TensorMol.ForceModifiers.NeighborsMB import *
from TensorMol.ForceModifiers.Periodic import *
from TensorMol.ForcesModels.Electrostatics import *
from TensorMol.ForcesModels.ElectrostaticsTF import *
from TensorMol.ForcesModels.TFPeriodicForces import *

from TensorMol.Simulations.Opt import *
from TensorMol.Simulations.Neb import *
from TensorMol.Simulations.SimpleMD import *
from TensorMol.Simulations.InfraredMD import *
from TensorMol.Simulations.MetaDynamics import *
from TensorMol.Simulations.OptPeriodic import *
from TensorMol.Simulations.PeriodicMD import *
from TensorMol.Simulations.PeriodicMC import *

from TensorMol.TFNetworks.TFBehlerParinello import *
from TensorMol.TFNetworks.TFBehlerParinelloSymEE import *
LOGGER.debug("TensorMol import complete.")
