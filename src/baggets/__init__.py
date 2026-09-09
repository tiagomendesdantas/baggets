"""baggets — bagged bootstrap ETS ensembles with pluggable selection.

Python reimplementation of Bagged.Cluster.ETS (Dantas & Cyrino Oliveira, 2018,
International Journal of Forecasting 34(4):748-761) plus a research platform for
ensemble-selection experiments.
"""

from .bootstrap import BootstrapResult, bld_mbb_bootstrap, moving_block_bootstrap
from .boxcox import boxcox, guerrero_lambda, inv_boxcox
from .decompose import decompose
from .engine import ETSEngine, ForecastEngine
from .model import BaggedETS, Forecast
from .nbeats import FittedNBeats, NBeatsConfig, NBeatsEngine
from .selection import (
    ClusterSelection,
    GreedyCovarianceSelection,
    NoSelection,
    PortfolioSelection,
    RandomSelection,
    SelectionStrategy,
    TopKSelection,
)
from .validation import ValidationArtifacts, build_validation_artifacts

__version__ = "0.1.0"

__all__ = [
    "BaggedETS",
    "Forecast",
    "NBeatsEngine",
    "NBeatsConfig",
    "FittedNBeats",
    "SelectionStrategy",
    "ClusterSelection",
    "GreedyCovarianceSelection",
    "PortfolioSelection",
    "TopKSelection",
    "RandomSelection",
    "NoSelection",
    "ETSEngine",
    "ForecastEngine",
    "ValidationArtifacts",
    "build_validation_artifacts",
    "BootstrapResult",
    "bld_mbb_bootstrap",
    "moving_block_bootstrap",
    "boxcox",
    "guerrero_lambda",
    "inv_boxcox",
    "decompose",
    "__version__",
]
