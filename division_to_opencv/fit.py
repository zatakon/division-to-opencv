"""Fitting OpenCV distortion models to a division model.

Both models are radial around the same centre, so the whole job is a 1-D curve fit of
r_d as a function of the undistorted geometry - no calibration images are needed.

* rational (cv2.initUndistortRectifyMap, ROS "rational_polynomial"):
      r_d = r_u (1 + k1 x + k2 x^2 + k3 x^3) / (1 + k4 x + k5 x^2 + k6 x^3),  x = (r_u / f)^2
  Pinhole based, so it can only cover angles below 90 deg.
* fisheye (cv2.fisheye, Kannala-Brandt / ROS "equidistant"):
      r_d = f theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
  Works with the angle, so the model also covers rays beyond 90 deg. Note that
  cv2.fisheye.projectPoints computes the angle from x/z and so cannot project those
  rays; use opencv.project_fisheye for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import DivisionModel


@dataclass
class FitResult:
    """Fitted OpenCV distortion and its accuracy over the fitted range.

    D is in OpenCV order: rational -> [k1, k2, p1, p2, k3, k4, k5, k6] (p1 = p2 = 0),
    fisheye -> [k1, k2, k3, k4]. Errors are in distorted-image pixels.
    """

    kind: str
    D: np.ndarray
    focal_length: float
    r_max: float            # fitted distorted radius [px]
    max_angle_deg: float    # angle off-axis covered by the fit
    max_error_px: float
    rms_error_px: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "model": self.kind,
            "D": [float(v) for v in self.D],
            "focal_length": self.focal_length,
            "fitted_radius_px": self.r_max,
            "fitted_angle_deg": self.max_angle_deg,
            "max_error_px": self.max_error_px,
            "rms_error_px": self.rms_error_px,
            "notes": list(self.notes),
        }


def max_radius(size, center):
    """Distance from the centre of distortion to the farthest image corner [px]."""
    w, h = size
    cx, cy = center
    return float(np.hypot(max(cx + 0.5, w - 0.5 - cx), max(cy + 0.5, h - 0.5 - cy)))


def _radius_at_angle(model, focal_length, angle_rad, r_max):
    """Smallest r_d <= r_max seen at angle_rad off-axis, or r_max when never reached."""
    r = np.linspace(0.0, r_max, 200001)
    over = model.angle(r, focal_length) >= angle_rad
    if not over.any():
        return r_max
    i = int(np.argmax(over))
    lo, hi = r[max(i - 1, 0)], r[i]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if model.angle(mid, focal_length) < angle_rad else (lo, mid)
    return 0.5 * (lo + hi)


def _rational_eval(k, r_u, f):
    x = (r_u / f) ** 2
    num = 1 + k[0] * x + k[1] * x**2 + k[2] * x**3
    den = 1 + k[3] * x + k[4] * x**2 + k[5] * x**3
    return r_u * num / den, num, den, x


def fit_rational(model: DivisionModel, focal_length, r_max, max_angle_deg=80.0, samples=4000, refine=True):
    """Fit the OpenCV rational model on r_d in (0, r_max], limited to max_angle_deg off-axis."""
    f = float(focal_length)
    notes = []
    r_fit = _radius_at_angle(model, f, np.radians(max_angle_deg), r_max)
    if r_fit < r_max:
        notes.append(f"limited to {max_angle_deg:g} deg off-axis: r_d <= {r_fit:.1f} px of {r_max:.1f} px")
    r_d = np.linspace(r_fit / samples, r_fit, samples)
    r_u = model.undistort_radius(r_d)
    if not np.all(np.diff(r_u) > 0):
        raise ValueError("the division model is not monotonic on the fitted range")

    # Linear least squares: multiplying out the denominator makes the model linear in k
    q = r_d / r_u
    x = (r_u / f) ** 2
    a = np.stack([x, x**2, x**3, -q * x, -q * x**2, -q * x**3], axis=1)
    k = np.linalg.lstsq(a, q - 1.0, rcond=None)[0]

    if refine:
        k = _refine_rational(k, r_u, r_d, f)

    pred, _, den, _ = _rational_eval(k, r_u, f)
    if np.any(den <= 0):
        raise ValueError("the fitted rational model has a pole inside the fitted range")
    err = pred - r_d
    angle = float(np.degrees(model.angle(r_fit, f)))
    D = np.array([k[0], k[1], 0.0, 0.0, k[2], k[3], k[4], k[5]])
    return FitResult("rational", D, f, float(r_fit), angle, float(np.max(np.abs(err))),
                     float(np.sqrt(np.mean(err**2))), notes)


def _refine_rational(k, r_u, r_d, f, iterations=100):
    """Levenberg-Marquardt on the residual in distorted pixels (the linear fit minimises a proxy)."""
    def residual(k):
        pred, _, den, _ = _rational_eval(k, r_u, f)
        return pred - r_d, den

    res, den = residual(k)
    cost = float(res @ res)
    lam = 1e-3
    for _ in range(iterations):
        pred, num, den, x = _rational_eval(k, r_u, f)
        cols = [r_u * x**i / den for i in (1, 2, 3)] + [-pred * x**i / den for i in (1, 2, 3)]
        j = np.stack(cols, axis=1)
        scale = np.linalg.norm(j, axis=0)
        scale[scale == 0] = 1.0
        js = j / scale
        g = js.T @ res
        h = js.T @ js
        old_cost = cost
        for _ in range(10):
            step = np.linalg.solve(h + lam * np.diag(np.diag(h)), -g) / scale
            new_res, new_den = residual(k + step)
            new_cost = float(new_res @ new_res)
            if np.all(new_den > 0) and new_cost < cost:
                k, res, cost = k + step, new_res, new_cost
                lam = max(lam / 3, 1e-12)
                break
            lam *= 4
        if cost >= old_cost * (1 - 1e-10):
            break
    return k


def fit_fisheye(model: DivisionModel, focal_length, r_max, max_angle_deg=None, samples=4000):
    """Fit the OpenCV fisheye model on r_d in (0, r_max]; rays beyond 90 deg are fine.

    max_angle_deg limits the fit to the lens' field of view (e.g. its image circle).
    """
    f = float(focal_length)
    notes = []
    if max_angle_deg is not None:
        r_fit = _radius_at_angle(model, f, np.radians(max_angle_deg), r_max)
        if r_fit < r_max:
            notes.append(f"limited to {max_angle_deg:g} deg off-axis: r_d <= {r_fit:.1f} px of {r_max:.1f} px")
        r_max = r_fit
    r_d = np.linspace(r_max / samples, r_max, samples)
    theta = model.angle(r_d, f)
    if not np.all(np.diff(theta) > 0):
        raise ValueError("the division model's view angle is not monotonic on the fitted range")
    # r_d / f = theta + k1 theta^3 + ... + k4 theta^9 is linear in k and already in pixel units
    a = np.stack([theta**3, theta**5, theta**7, theta**9], axis=1)
    k = np.linalg.lstsq(a, r_d / f - theta, rcond=None)[0]
    pred = f * theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    err = pred - r_d
    return FitResult("fisheye", k, f, float(r_max), float(np.degrees(theta[-1])),
                     float(np.max(np.abs(err))), float(np.sqrt(np.mean(err**2))), notes)
