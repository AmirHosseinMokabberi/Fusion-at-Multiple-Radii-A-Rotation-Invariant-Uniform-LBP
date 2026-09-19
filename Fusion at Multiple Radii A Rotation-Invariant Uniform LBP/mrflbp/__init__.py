"""Multi-Radius Fused rotation-invariant uniform LBP for finger-vein identification.

Reference implementation for:

    A. Mokabberi and O. Toygar, "Fusion at multiple radii: a rotation-invariant
    uniform LBP for finger-vein identification", Signal, Image and Video
    Processing, 19:1419, 2025. https://doi.org/10.1007/s11760-025-05009-3

The package implements the descriptor, the dimensionality-reduction baselines,
the two evaluation strategies and the three train/test protocols reported in
the paper.
"""

from mrflbp.descriptors import (
    FUSED_CONFIGS,
    MultiRadiusLBP,
    riu2_mapping,
)
from mrflbp.evaluation import cmc_curve, mcnemar_test, rank1_accuracy
from mrflbp.matching import NearestNeighbourMatcher
from mrflbp.protocols import Evaluation, Protocol, Strategy
from mrflbp.reduction import PCA, TwoDimensionalPCA, TwoDirectionalTwoDimensionalPCA

__version__ = "1.0.0"

__all__ = [
    "FUSED_CONFIGS",
    "MultiRadiusLBP",
    "riu2_mapping",
    "PCA",
    "TwoDimensionalPCA",
    "TwoDirectionalTwoDimensionalPCA",
    "NearestNeighbourMatcher",
    "Strategy",
    "Protocol",
    "Evaluation",
    "rank1_accuracy",
    "cmc_curve",
    "mcnemar_test",
    "__version__",
]
