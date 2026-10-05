"""Radial division distortion model.

The division model (Fitzgibbon 2001) maps a distorted radius r_d, measured in pixels
from the centre of distortion, to the undistorted radius

    r_u = r_d / P(r_d),    P(r) = 1 + c1 r^2 + c2 r^4 + ... + cn r^(2n)

so undistortion is closed form and distortion needs an inverse. Seen as a ray, a
distorted pixel at radius r_d looks along (r_d, f P(r_d)), i.e. at the angle

    theta = atan2(r_d, f P(r_d))

from the optical axis. P(r_d) = 0 is 90 deg off-axis; beyond it (P < 0) the ray points
backwards, which a fisheye lens can see but a pinhole image cannot show.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DivisionModel:
    """Division model with polynomial coefficients c1..cn (pixel units)."""

    coeffs: tuple[float, ...]

    def __init__(self, coeffs):
        object.__setattr__(self, "coeffs", tuple(float(c) for c in coeffs))

    def ratio(self, r_d):
        """P(r_d) = 1 + sum c_i r_d^(2i)."""
        r_d = np.asarray(r_d, dtype=np.float64)
        r2 = r_d * r_d
        val = np.zeros_like(r_d)
        for c in reversed(self.coeffs):
            val = (val + c) * r2
        return 1.0 + val

    def undistort_radius(self, r_d):
        """r_u = r_d / P(r_d); inf where the point is 90 deg or more off-axis."""
        r_d = np.asarray(r_d, dtype=np.float64)
        p = self.ratio(r_d)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p > 0, r_d / p, np.inf)

    def angle(self, r_d, focal_length):
        """Angle from the optical axis [rad] of the ray seen at distorted radius r_d."""
        r_d = np.asarray(r_d, dtype=np.float64)
        return np.arctan2(r_d, focal_length * self.ratio(r_d))

    def radius_at_90deg(self, r_max):
        """Smallest r_d <= r_max with P(r_d) <= 0, or None when the model stays in front."""
        r = np.linspace(0.0, r_max, 200001)
        bad = self.ratio(r) <= 0
        if not bad.any():
            return None
        i = int(np.argmax(bad))
        lo, hi = r[i - 1], r[i]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.ratio(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    def distort_radius(self, r_u, r_max):
        """Inverse of undistort_radius by bisection on [0, r_max] (r_u must be reachable)."""
        r_u = np.asarray(r_u, dtype=np.float64)
        r90 = self.radius_at_90deg(r_max)
        hi_lim = r90 if r90 is not None else r_max
        lo = np.zeros_like(r_u)
        hi = np.full_like(r_u, hi_lim)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self.undistort_radius(mid) < r_u
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)
