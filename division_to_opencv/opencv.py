"""OpenCV side: camera matrices, undistortion maps and an exact division-model reference.

Pixel convention is OpenCV's: pixel centres at integer coordinates. Square pixels
(fx = fy) are assumed, as the division model is defined on a pixel radius.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fit import FitResult
from .model import DivisionModel


def camera_matrix(focal_length, center):
    """3x3 intrinsic matrix with fx = fy = focal_length and principal point at center."""
    f = float(focal_length)
    return np.array([[f, 0.0, center[0]], [0.0, f, center[1]], [0.0, 0.0, 1.0]])


def undistortion_maps(fit: FitResult, K, newK, size):
    """map_x, map_y (float32) for cv2.remap from the fitted model, output size = (w, h)."""
    if fit.kind == "rational":
        return cv2.initUndistortRectifyMap(K, fit.D, None, newK, tuple(size), cv2.CV_32FC1)
    if fit.kind == "fisheye":
        return cv2.fisheye.initUndistortRectifyMap(K, fit.D, np.eye(3), newK, tuple(size), cv2.CV_32FC1)
    raise ValueError(f"unknown model {fit.kind!r}")


def project_fisheye(points, K, D):
    """Project 3-D points (N x 3, camera frame) with the fisheye (Kannala-Brandt) model.

    Same formula as cv2.fisheye.projectPoints, but the angle is atan2(sqrt(x^2 + y^2), z),
    so points more than 90 deg off-axis (z <= 0) project correctly too; OpenCV's version
    divides by z and folds them back in front of the camera.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r = np.hypot(p[:, 0], p[:, 1])
    theta = np.arctan2(r, p[:, 2])
    k1, k2, k3, k4 = np.asarray(D, dtype=np.float64).ravel()[:4]
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(r > 0, theta_d / r, 0.0)
    u = K[0, 0] * p[:, 0] * s + K[0, 1] * p[:, 1] * s + K[0, 2]
    v = K[1, 1] * p[:, 1] * s + K[1, 2]
    return np.stack([u, v], axis=1)


def reference_maps(model: DivisionModel, K, newK, size, r_max):
    """Exact remap maps of the division model for the same K / newK (NaN where undefined).

    For every output pixel the undistorted radius in distorted-image pixels is
    r_u = f |(u - c') / f'|, the distorted radius solves r_d / P(r_d) = r_u, and the
    source pixel is c + direction * r_d.
    """
    w, h = size
    f = K[0, 0]
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dx = (u - newK[0, 2]) / newK[0, 0] * f
    dy = (v - newK[1, 2]) / newK[1, 1] * f
    r_u = np.hypot(dx, dy)
    r_d = model.distort_radius(r_u.ravel(), r_max).reshape(r_u.shape)
    reachable = np.abs(model.undistort_radius(r_d) - r_u) <= 1e-6 * np.maximum(r_u, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(r_u > 0, r_d / r_u, 1.0)
    map_x = np.where(reachable, K[0, 2] + dx * s, np.nan)
    map_y = np.where(reachable, K[1, 2] + dy * s, np.nan)
    return map_x, map_y


def map_error(fit_maps, ref_maps, src_size=None):
    """Per-pixel distance between two maps [px]; NaN where the reference is undefined
    or (with src_size) where it samples outside the source image."""
    mx, my = fit_maps
    rx, ry = ref_maps
    err = np.hypot(mx - rx, my - ry)
    if src_size is not None:
        w, h = src_size
        inside = (rx >= -0.5) & (rx <= w - 0.5) & (ry >= -0.5) & (ry <= h - 0.5)
        err = np.where(inside, err, np.nan)
    return err
