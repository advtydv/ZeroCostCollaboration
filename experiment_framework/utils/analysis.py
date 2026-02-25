"""Backward-compatible analysis module.

This module previously contained a broken implementation with indentation issues.
For compatibility, it now re-exports the maintained implementation.
"""

from .analysis_fixed import ExperimentAnalyzer

__all__ = ["ExperimentAnalyzer"]
