"""Applying a 3x3 transformation to an image.

Forward mapping  : push each INPUT pixel to M @ p. Leaves holes.
Inverse mapping  : for each OUTPUT pixel, look back at M^-1 @ q in the input
                   and interpolate. Every output pixel is filled. (Used by the app.)
numpy is used only as a fast container for pixel arrays; the matrix maths
(M, inverse, composition) comes from matrix.py / transforms.py.
"""
import math

import numpy as np

from matrix import inverse, mat_vec
from transforms import compose, translation
MAX_SIDE = 4000  # refuse absurdly large output canvases


def output_canvas(M, width, height, expand=True):
    """Return (out_w, out_h, M_final). With expand=True the canvas is the
    bounding box of the transformed image corners, shifted to start at (0, 0)."""
    if not expand:
        return width, height, M
    xs, ys = [], []
    for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
        p = mat_vec(M, [x, y, 1.0])
        xs.append(p[0])
        ys.append(p[1])
    min_x, min_y = min(xs), min(ys)
    out_w = int(math.ceil(max(xs) - min_x - 1e-9)) + 1
    out_h = int(math.ceil(max(ys) - min_y - 1e-9)) + 1
    if out_w > MAX_SIDE or out_h > MAX_SIDE:
        raise ValueError(f"Result would be {out_w}x{out_h} pixels, larger than "
                         f"the {MAX_SIDE}px limit. Use a smaller scale.")
    return out_w, out_h, compose([M, translation(-min_x, -min_y)])


def _source_coords(Minv, out_w, out_h):
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    xs = xs.astype(float)
    ys = ys.astype(float)
    xi = Minv[0][0] * xs + Minv[0][1] * ys + Minv[0][2]
    yi = Minv[1][0] * xs + Minv[1][1] * ys + Minv[1][2]
    return xi, yi


def inverse_warp(img, M, interpolation="bilinear", expand=True, fill=0,
                 return_mask=False):
    """Warp img by M using inverse mapping.

    Raises SingularMatrixError when det(M) = 0 (no inverse exists).
    """
    h, w = img.shape[:2]
    out_w, out_h, Mf = output_canvas(M, w, h, expand)
    Minv = inverse(Mf)
    xi, yi = _source_coords(Minv, out_w, out_h)
    out = np.full((out_h, out_w) + img.shape[2:], fill, dtype=img.dtype)

    if interpolation == "nearest":
        valid = (xi >= -0.5) & (xi < w - 0.5) & (yi >= -0.5) & (yi < h - 0.5)
        xr = np.clip(np.rint(xi).astype(int), 0, w - 1)
        yr = np.clip(np.rint(yi).astype(int), 0, h - 1)
        out[valid] = img[yr[valid], xr[valid]]
    elif interpolation == "bilinear":
        eps = 1e-6
        valid = (xi >= -eps) & (xi <= w - 1 + eps) & (yi >= -eps) & (yi <= h - 1 + eps)
        x = np.clip(xi, 0, w - 1)
        y = np.clip(yi, 0, h - 1)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        if img.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        f = img.astype(float)
        top = f[y0, x0] * (1 - fx) + f[y0, x1] * fx
        bot = f[y1, x0] * (1 - fx) + f[y1, x1] * fx
        val = top * (1 - fy) + bot * fy
        out[valid] = np.clip(np.rint(val[valid]), 0, 255).astype(img.dtype)
    else:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")

    return (out, valid) if return_mask else out


def forward_warp(img, M, expand=True, fill=0):
    """Naive forward mapping, kept to SHOW the holes it leaves.
    Works even when M is singular (the image collapses onto a line)."""
    h, w = img.shape[:2]
    out_w, out_h, Mf = output_canvas(M, w, h, expand)
    ys, xs = np.mgrid[0:h, 0:w]
    xs_f, ys_f = xs.astype(float), ys.astype(float)
    xo = np.rint(Mf[0][0] * xs_f + Mf[0][1] * ys_f + Mf[0][2]).astype(int)
    yo = np.rint(Mf[1][0] * xs_f + Mf[1][1] * ys_f + Mf[1][2]).astype(int)
    ok = (xo >= 0) & (xo < out_w) & (yo >= 0) & (yo < out_h)
    out = np.full((out_h, out_w) + img.shape[2:], fill, dtype=img.dtype)
    written = np.zeros((out_h, out_w), dtype=bool)
    out[yo[ok], xo[ok]] = img[ys[ok], xs[ok]]
    written[yo[ok], xo[ok]] = True
    return out, written


def forward_vs_inverse(img, M, expand=True):
    """Compare both methods; holes = pixels inside the image footprint that
    forward mapping never wrote."""
    inv_img, footprint = inverse_warp(img, M, "nearest", expand, return_mask=True)
    fwd_img, written = forward_warp(img, M, expand)
    holes = footprint & ~written
    return {"forward": fwd_img, "inverse": inv_img, "holes": holes,
            "n_holes": int(holes.sum()), "n_footprint": int(footprint.sum())}
