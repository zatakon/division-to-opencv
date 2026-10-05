import json

import numpy as np
import pytest

from division_to_opencv import (
    DivisionModel, camera_matrix, fit_fisheye, fit_rational, map_error, max_radius,
    project_fisheye, reference_maps, undistortion_maps,
)
from division_to_opencv.cli import main

# Made-up cameras, not taken from any real calibration
WIDE = dict(size=(1920, 1080), center=(963.4, 538.2), f=1050.0, c=(-2.1e-7, -4.0e-14))
FISHEYE = dict(size=(1280, 960), center=(641.7, 482.3), f=310.0, c=(-4.4e-6, -6.0e-12), circle_deg=100.0)


def test_radius_roundtrip():
    m = DivisionModel(WIDE["c"])
    r_d = np.linspace(0.0, 1100.0, 1000)
    np.testing.assert_allclose(m.distort_radius(m.undistort_radius(r_d), 1200.0), r_d, atol=1e-6)


def test_radius_at_90deg():
    m = DivisionModel((-1.0e-6,))
    assert m.radius_at_90deg(2000.0) == pytest.approx(1000.0, abs=1e-6)
    assert DivisionModel((1.0e-7,)).radius_at_90deg(2000.0) is None
    assert m.angle(1000.0, 500.0) == pytest.approx(np.pi / 2)


def test_rational_fit_covers_the_corners():
    m = DivisionModel(WIDE["c"])
    r_max = max_radius(WIDE["size"], WIDE["center"])
    fit = fit_rational(m, WIDE["f"], r_max)
    assert fit.r_max == pytest.approx(r_max)
    assert fit.max_error_px < 0.01
    assert fit.D.shape == (8,) and fit.D[2] == 0 and fit.D[3] == 0


def test_rational_maps_match_the_division_model():
    m = DivisionModel(WIDE["c"])
    r_max = max_radius(WIDE["size"], WIDE["center"])
    K = camera_matrix(WIDE["f"], WIDE["center"])
    newK = camera_matrix(WIDE["f"] / 2, (480.0, 270.0))
    size = (960, 540)
    ref = reference_maps(m, K, newK, size, r_max)
    for fit in (fit_rational(m, WIDE["f"], r_max), fit_fisheye(m, WIDE["f"], r_max)):
        err = map_error(undistortion_maps(fit, K, newK, size), ref, WIDE["size"])
        assert np.isfinite(err).mean() > 0.9
        assert np.nanmax(err) < 0.02, fit.kind


def test_rational_stops_before_90deg():
    m = DivisionModel(FISHEYE["c"])
    r_max = max_radius(FISHEYE["size"], FISHEYE["center"])
    fit = fit_rational(m, FISHEYE["f"], r_max, max_angle_deg=80.0)
    assert fit.max_angle_deg == pytest.approx(80.0, abs=0.01)
    assert fit.r_max < m.radius_at_90deg(r_max)
    assert fit.notes and fit.max_error_px < 0.2


def test_fisheye_beyond_90deg():
    m = DivisionModel(FISHEYE["c"])
    r_max = max_radius(FISHEYE["size"], FISHEYE["center"])
    fit = fit_fisheye(m, FISHEYE["f"], r_max, max_angle_deg=FISHEYE["circle_deg"])
    assert fit.max_angle_deg == pytest.approx(FISHEYE["circle_deg"], abs=0.01)
    assert fit.max_error_px < 0.1

    # rays up to the image circle (beyond 90 deg) land where the division model sees them
    K = camera_matrix(FISHEYE["f"], FISHEYE["center"])
    r_d = np.linspace(10.0, fit.r_max, 50)
    theta = m.angle(r_d, FISHEYE["f"])
    rays = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)  # along +x
    pts = project_fisheye(rays, K, fit.D)
    np.testing.assert_allclose(pts[:, 0] - FISHEYE["center"][0], r_d, atol=0.1)
    np.testing.assert_allclose(pts[:, 1], FISHEYE["center"][1], atol=1e-9)

    # in front of the camera it agrees with OpenCV's own projection
    import cv2
    front = theta < np.radians(85)
    cv_pts, _ = cv2.fisheye.projectPoints(rays[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, fit.D)
    np.testing.assert_allclose(cv_pts[:, 0, :], pts[front], atol=1e-6)


def test_non_monotonic_model_is_rejected():
    m = DivisionModel((5.0e-6,))  # r_u = r_d / (1 + c r_d^2) turns back at r_d = 447 px
    with pytest.raises(ValueError):
        fit_rational(m, 500.0, 800.0)


def test_cli_json(capsys):
    c = [str(v) for v in WIDE["c"]]
    assert main(["--coeffs", *c, "--focal", "1050", "--center", "963.4", "538.2",
                 "--size", "1920", "1080", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["fits"]) == {"rational", "fisheye"}
    assert out["fits"]["rational"]["max_error_px"] < 0.01
    assert out["K"][0][2] == pytest.approx(963.4)
