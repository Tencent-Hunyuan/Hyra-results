"""HyPDE: a 2D foundation model for partial differential equations."""
from .api import Forecaster, load
from .data import DATASETS, DatasetInfo, Trajectory, dataset, load_trajectory
from .model import HyPDE, HyPDEConfig

__version__ = "0.1.0"
__all__ = ["Forecaster", "load", "HyPDE", "HyPDEConfig", "DATASETS", "DatasetInfo", "dataset", "Trajectory",
           "load_trajectory", "__version__"]
