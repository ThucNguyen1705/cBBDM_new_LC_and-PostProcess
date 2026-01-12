"""
Refinement Module Package
=========================

Phase 2: Texture Refinement Module for SAR-to-Optical translation.

Modules:
    - TextureRefinementModule: Full version with encoder-decoder + MARM
    - TextureRefinementModuleLite: Lightweight version for faster training
"""

from .refinement import TextureRefinementModule, TextureRefinementModuleLite

__all__ = ['TextureRefinementModule', 'TextureRefinementModuleLite']
