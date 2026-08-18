"""GPU-native RIS-assisted MIMO statistical environment.

Core scope: MATLAB-parity-validated deterministic statistics, stochastic
channel generation, Type-I rank-1 codebooks, RIS response, and symmetric
Gamma-Gamma q05 label generation.
"""

from .antenna import ArraySpec, generate_channel_moments_batch
from .geometry_lsp import generate_geometry_batch, generate_lsp_batch
from .ris_response import generate_ris_response_from_z
from .codebook import generate_codebook_rank1
from .environment import BankInput, build_deterministic_bank, evaluate_z_candidates
from .gamma_gamma import GGQ05Lookup, symmetric_gg_q05_numpy
from .label_engine import run_symmetric_gg_label_engine

__all__ = [
    "ArraySpec",
    "generate_channel_moments_batch",
    "generate_geometry_batch",
    "generate_lsp_batch",
    "generate_ris_response_from_z",
    "generate_codebook_rank1",
    "BankInput",
    "build_deterministic_bank",
    "evaluate_z_candidates",
    "GGQ05Lookup",
    "symmetric_gg_q05_numpy",
    "run_symmetric_gg_label_engine",
]
