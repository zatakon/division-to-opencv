"""Demo on two made-up cameras: writes the README figures to docs/.

    python examples/demo.py
"""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from division_to_opencv import (  # noqa: E402
    DivisionModel, camera_matrix, fit_fisheye, fit_rational, map_error, max_radius,
    reference_maps, undistortion_maps,
)

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
C_RAT, C_FISH = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

CAMERAS = {
    "wide": dict(title="Wide-angle camera, 1920x1080, f = 1050 px",
                 size=(1920, 1080), center=(963.4, 538.2), f=1050.0, c=(-2.1e-7, -4.0e-14),
                 out_size=(960, 540), out_f=525.0, square=0.12, figsize=(14, 6.4)),
    "fisheye": dict(title="Fisheye camera, 1280x960, f = 310 px, 200 deg image circle",
                    size=(1280, 960), center=(641.7, 482.3), f=310.0, c=(-4.4e-6, -6.0e-12),
                    circle_deg=100.0, out_size=(720, 720), out_f=131.0, square=0.25, figsize=(14, 8.6)),
}


def synthetic_image(model, cam):
    """Checkerboard on a plane in front of the camera, seen through the division model."""
    w, h = cam["size"]
    u, v = np.meshgrid(np.arange(w) - cam["center"][0], np.arange(h) - cam["center"][1])
    r_d = np.hypot(u, v)
    p = model.ratio(r_d)
    front = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(front, u / (cam["f"] * p), 0)
        y = np.where(front, v / (cam["f"] * p), 0)
    sq = cam["square"]
    img = 70 + 110 * ((np.floor(x / sq) + np.floor(y / sq)) % 2)
    img[~front] = 40                       # 90 deg or more off-axis: behind the plane
    if "circle_deg" in cam:
        img[model.angle(r_d, cam["f"]) > np.radians(cam["circle_deg"])] = 0
    return img.astype(np.uint8)


def remap(img, maps):
    mx, my = (np.nan_to_num(m, nan=-10.0).astype(np.float32) for m in maps)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def clean(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def run(name, cam):
    model = DivisionModel(cam["c"])
    r_max = max_radius(cam["size"], cam["center"])
    rat = fit_rational(model, cam["f"], r_max)
    fish = fit_fisheye(model, cam["f"], r_max, cam.get("circle_deg"))
    K = camera_matrix(cam["f"], cam["center"])
    ow, oh = cam["out_size"]
    newK = camera_matrix(cam["out_f"], ((ow - 1) / 2, (oh - 1) / 2))
    ref = reference_maps(model, K, newK, cam["out_size"], r_max)
    rat_maps = undistortion_maps(rat, K, newK, cam["out_size"])
    err = map_error(rat_maps, ref, cam["size"])

    src = synthetic_image(model, cam)
    und_ref = remap(src, ref)
    und_cv = remap(src, rat_maps)
    diff = np.abs(und_ref.astype(np.int16) - und_cv.astype(np.int16)).astype(float)
    diff[~np.isfinite(err)] = np.nan

    fig = plt.figure(figsize=cam["figsize"])
    gs = fig.add_gridspec(2, 3)
    panels = [
        (src, f"Input {cam['size'][0]}x{cam['size'][1]} (synthetic)"),
        (und_ref, "Undistorted: division model (exact)"),
        (und_cv, "Undistorted: cv2.initUndistortRectifyMap"),
    ]
    for i, (im, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        clean(ax)

    ax = fig.add_subplot(gs[1, 0])
    r = np.linspace(1.0, r_max, 3000)
    theta = model.angle(r, cam["f"])
    for fit, color in ((rat, C_RAT), (fish, C_FISH)):
        sel = r <= fit.r_max
        t = theta[sel]
        if fit.kind == "rational":
            x = (np.tan(t)) ** 2
            D = fit.D
            pred = cam["f"] * np.tan(t) * (1 + D[0] * x + D[1] * x**2 + D[4] * x**3) / (1 + D[5] * x + D[6] * x**2 + D[7] * x**3)
        else:
            k = fit.D
            pred = cam["f"] * t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
        ax.semilogy(np.degrees(t), np.abs(pred - r[sel]) + 1e-7, color=color, lw=2,
                    label=f"{fit.kind}: max {fit.max_error_px:.2g} px")
    ax.axhline(0.1, color=INK2, lw=1, ls=":")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("angle off the optical axis [deg]")
    ax.set_ylabel("|r_d error| [px]")
    ax.set_title("Fit error along the radius\n ")
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(err, cmap="Blues", vmin=0, vmax=np.nanmax(err))
    ax.set_title(f"Remap error (rational)\nmax {np.nanmax(err):.2g} px")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="px in the input image")
    clean(ax)

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(diff, cmap="Blues", vmin=0, vmax=max(8.0, np.nanmax(diff)))
    ax.set_title(f"|exact - OpenCV| image difference\nmean {np.nanmean(diff):.2g} grey levels")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="grey levels (0-255)")
    clean(ax)

    fig.suptitle(cam["title"], x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(DOCS, f"{name}.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"{name}: rational max {rat.max_error_px:.4f} px (to {rat.max_angle_deg:.0f} deg), "
          f"fisheye max {fish.max_error_px:.4f} px (to {fish.max_angle_deg:.0f} deg), "
          f"remap max {np.nanmax(err):.4f} px -> {os.path.normpath(path)}")


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    for name, cam in CAMERAS.items():
        run(name, cam)
