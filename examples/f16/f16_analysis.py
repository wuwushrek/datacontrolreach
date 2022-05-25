from jax.config import config
config.update("jax_enable_x64", True)

import jax

from datacontrolreach import jumpy as jp
import numpy as np

from datacontrolreach import interval
from datacontrolreach.interval import Interval as iv

from datacontrolreach import dynsideinfo
from datacontrolreach.dynsideinfo import DynamicsWithSideInfo, \
	lipinfo_builder, hc4revise_lin_eq

from datacontrolreach import reach
from datacontrolreach.reach import build_overapprox, DaTaReach

from tqdm.auto import tqdm

####### Let's load trajectories fron the F-16 simulator ##########



####### Build the F-16 model with side information ###############
lipinfo = {}
nS, nC = 12, 4
# Index order in the state
_VT, _ALPHA, _BETA, _ROLL, _PITCH, _P, _Q, _R, _H, _POW, _iNZ, _iPs = \
	[i for i in range(nS)]
# vt : 0, alpha : 1, beta : 2, roll : 3, pitch : 4, p : 5, q : 6, r : 7
# alt : 8, power : 9, Int Nz : 10, int Ps : 11

# Thrust unknown terms
lipinfo['_thrust0'] = lipinfo_builder(Lip=1.1, vDep=[_H,_VT], weightLip=None, n=nS,
                    					bound=(-1000,0), gradBound=None)