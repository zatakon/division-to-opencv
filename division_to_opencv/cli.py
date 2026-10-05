"""Command line: fit OpenCV distortion coefficients to a division model."""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from .fit import fit_fisheye, fit_rational, max_radius
from .model import DivisionModel


def build_parser():
    p = argparse.ArgumentParser(
        prog="division-to-opencv",
        description="Fit OpenCV distortion coefficients (rational / fisheye) to a division model "
                    "r_u = r_d / (1 + c1 r_d^2 + c2 r_d^4 + ...), radii in pixels.",
    )
    p.add_argument("--coeffs", type=float, nargs="+", required=True, metavar="C",
                   help="division model coefficients c1 c2 ... (pixel units)")
    p.add_argument("--focal", type=float, required=True, help="focal length [px] (fx = fy)")
    p.add_argument("--center", type=float, nargs=2, metavar=("CX", "CY"),
                   help="centre of distortion [px, OpenCV convention: pixel centres at integers]")
    p.add_argument("--size", type=int, nargs=2, metavar=("W", "H"),
                   help="image size; the fit then covers the farthest image corner")
    p.add_argument("--max-radius", type=float, help="fit up to this distorted radius [px] instead of --size")
    p.add_argument("--model", choices=["rational", "fisheye", "both"], default="both")
    p.add_argument("--max-angle", type=float, default=80.0,
                   help="rational model: largest off-axis angle to fit [deg] (default 80, must be < 90)")
    p.add_argument("--fisheye-max-angle", type=float,
                   help="fisheye model: largest off-axis angle to fit [deg], e.g. the image circle (default: no limit)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    # argparse only takes "-1.5" as a negative number, not "-2.1e-07"; coefficients usually look like that
    p._negative_number_matcher = re.compile(r"^-(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_radius is not None:
        r_max = args.max_radius
    elif args.size is not None and args.center is not None:
        r_max = max_radius(args.size, args.center)
    else:
        sys.exit("give --max-radius, or --size together with --center")

    model = DivisionModel(args.coeffs)
    kinds = ["rational", "fisheye"] if args.model == "both" else [args.model]
    out = {"division_coeffs": list(model.coeffs), "focal_length": args.focal, "max_radius_px": r_max}
    if args.center is not None:
        cx, cy = args.center
        out["K"] = [[args.focal, 0.0, cx], [0.0, args.focal, cy], [0.0, 0.0, 1.0]]
    r90 = model.radius_at_90deg(r_max)
    if r90 is not None:
        out["radius_at_90deg_px"] = r90

    results = {}
    for kind in kinds:
        try:
            fit = fit_rational(model, args.focal, r_max, args.max_angle) if kind == "rational" \
                else fit_fisheye(model, args.focal, r_max, args.fisheye_max_angle)
            results[kind] = fit.to_dict()
        except ValueError as e:
            results[kind] = {"model": kind, "error": str(e)}
    out["fits"] = results

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"division model c = {list(model.coeffs)}, f = {args.focal:g} px, fitted up to r_d = {r_max:.1f} px")
    if r90 is not None:
        print(f"note: r_d >= {r90:.1f} px is 90 deg or more off-axis (no pinhole equivalent)")
    if "K" in out:
        print("K =", np.array(out["K"]).tolist())
    for kind, res in results.items():
        print(f"\n[{kind}]")
        if "error" in res:
            print("  failed:", res["error"])
            continue
        print("  D =", [float(f"{v:.9g}") for v in res["D"]])
        print(f"  fitted r_d <= {res['fitted_radius_px']:.1f} px ({res['fitted_angle_deg']:.1f} deg off-axis)")
        print(f"  error: max {res['max_error_px']:.4f} px, rms {res['rms_error_px']:.4f} px")
        for n in res["notes"]:
            print("  note:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
