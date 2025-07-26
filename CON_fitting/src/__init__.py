"""
CON_fitting: Consensus CNA Signature Fitting Framework

A robust framework for fitting consensus Copy Number Alteration (CNA) signatures
to new sample data, developed for reproducible research and clinical applications.

Author: Consensus CNA Signatures Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Consensus CNA Signatures Team"

from .signature_fitter import ConsensusSignatureFitter
from .data_processor import DataProcessor
from .visualizer import SignatureVisualizer
from .validator import SignatureValidator

__all__ = [
    'ConsensusSignatureFitter',
    'DataProcessor', 
    'SignatureVisualizer',
    'SignatureValidator'
] 