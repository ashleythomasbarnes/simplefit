"""Simple spectrum and cube fitting helpers."""

from .cube_fitting import FitCubeResult, fit_cube
from .spectrum_fitting import (
    FitResult,
    GaussianComponent,
    SpectrumInitialGuess,
    detect_initial_components,
    fit_spectrum,
    fit_spectrum_from_components,
    plot_fit,
)

__all__ = [
    "FitCubeResult",
    "FitResult",
    "GaussianComponent",
    "SpectrumInitialGuess",
    "detect_initial_components",
    "fit_cube",
    "fit_spectrum",
    "fit_spectrum_from_components",
    "plot_fit",
]
