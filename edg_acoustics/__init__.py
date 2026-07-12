"""Documentation about edg_acoustics (in init.py file)."""

import logging
from .acoustics_simulation import AcousticsSimulation
from .mesh import Mesh
from .boundary_condition import BoundaryCondition, AbsorbBC
from .initial_condition import InitialCondition, Monopole_IC, RadialPressurePulse2D_IC
from .preprocessing import Flux, UpwindFlux, MaterialUpwindFlux2D
from .time_integration import TimeIntegrator, TSI_TI
from .postprocessing import Monopole_postprocessor
from .mesh2d import Mesh2D
from .acoustics_simulation_2d import AcousticsSimulation2D
from .acoustics_simulation_2d_er import (
    ExtendedReactionMaterialFit,
    ExtendedReactionSimulation2D,
    VectorFittedSISO,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Huiqing Wang, Artur Palha"
__email__ = "h.wang6@tue.nl, A.Palha@tudelft.nl"
__version__ = "1.0.0-alpha.1"
