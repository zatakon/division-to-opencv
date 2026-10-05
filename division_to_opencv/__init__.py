"""Fit OpenCV distortion coefficients that reproduce a division-model undistortion."""
from .fit import FitResult, fit_fisheye, fit_rational, max_radius
from .model import DivisionModel
from .opencv import camera_matrix, map_error, project_fisheye, reference_maps, undistortion_maps

__all__ = [
    "DivisionModel",
    "FitResult",
    "fit_rational",
    "fit_fisheye",
    "max_radius",
    "camera_matrix",
    "undistortion_maps",
    "project_fisheye",
    "reference_maps",
    "map_error",
]
__version__ = "0.1.0"
