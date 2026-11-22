"""Generates the three required outputs + the validation table.

    python demo_outputs.py

Creates report/figures/*.png and tests/validation_results.md
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.imageio import make_sample_image, to_rgb
from app.session import EditorSession
from db.database import Database
from math_engine.matrix import SingularMatrixError, determinant, format_matrix, inverse, mat_mul
from math_engine.transforms import about_point, compose, describe_determinant, rotation, scaling
from math_engine.warp import forward_vs_inverse, forward_warp, inverse_warp

FIG = os.path.join("report", "figures")
os.makedirs(FIG, exist_ok=True)
img = make_sample_image()
h, w = img.shape[:2]
db = Database(os.path.join(tempfile.mkdtemp(), "demo.db"))


def show(ax, a, title):
    ax.imshow(a, cmap="gray" if a.ndim == 2 else None)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# 1. Chain of transformations, then undo ---------------------------------------
ed = EditorSession(db, img, "sample.png")
states = [("Original", ed.render())]
for name, p in [("Rotate", {"angle": 30}), ("Shear", {"kx": 0.4, "ky": 0}),
                ("Flip horizontal", {}), ("Scale", {"sx": 0.7, "sy": 0.7})]:
    ed.apply(name, p)
    states.append((f"+ {name}", ed.render()))
undone = []
for _ in range(4):
    ed.undo()
    undone.append(ed.render())
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for ax, (t, a) in zip(axs[0], states):
    show(ax, a, t)
for ax, t, a in zip(axs[1][:4], ["Undo 1", "Undo 2", "Undo 3", "Undo 4"], undone):
    show(ax, a, t)
axs[1][4].axis("off")
axs[1][4].text(0, 0.5, f"After 4 undos the image is\nidentical to the original:\n{np.array_equal(undone[-1], img)}", fontsize=10)
fig.suptitle("Output 1: chain of edits (top) and undo (bottom)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_chain_and_undo.png"), dpi=110); plt.close(fig)

# 2. Zero determinant -----------------------------------------------------------
M0 = about_point([[1, 0, 0], [0, 0, 0], [0, 0, 1]], (w - 1) / 2, (h - 1) / 2)
d0 = describe_determinant(M0)
fwd, _ = forward_warp(img, M0, expand=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
show(axs[0], img, "Original")
show(axs[1], fwd, f"Scale y by 0 (det = {d0['det']:.0f}): everything lands on one line")
try:
    inverse_warp(img, M0)
    msg = "unexpected"
except SingularMatrixError as e:
    msg = str(e)
fig.suptitle("Output 2: determinant = 0\n" + msg, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_zero_determinant.png"), dpi=110); plt.close(fig)

# 3. Why inverse mapping avoids holes ---------------------------------------------
M = about_point(compose([rotation(30), scaling(1.5, 1.5)]), (w - 1) / 2, (h - 1) / 2)
r = forward_vs_inverse(img, M)
marked = to_rgb(r["forward"]).copy()
marked[r["holes"]] = (255, 0, 0)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
show(axs[0], r["forward"], "Forward mapping (push pixels)")
show(axs[1], marked, f"Holes in red: {r['n_holes']:,} ({100*r['n_holes']/r['n_footprint']:.1f}%)")
show(axs[2], r["inverse"], "Inverse mapping: 0 holes")
fig.suptitle("Output 3: rotate 30 deg + scale 1.5")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_inverse_mapping_no_holes.png"), dpi=110); plt.close(fig)

# Validation table vs numpy -------------------------------------------------------------
rng = np.random.default_rng(42)
rows = []
for n in (2, 3, 5, 8):
    A = rng.normal(size=(n, n))
    B = rng.normal(size=(n, n))
    rows.append((n,
                 abs(determinant(A.tolist()) - np.linalg.det(A)),
                 np.abs(np.array(inverse(A.tolist())) - np.linalg.inv(A)).max(),
                 np.abs(np.array(mat_mul(A.tolist(), B.tolist())) - A @ B).max()))
with open(os.path.join("tests", "validation_results.md"), "w") as f:
    f.write("# Validation of from-scratch code against numpy\n\n")
    f.write("| n | abs error in det | max abs error in inverse | max abs error in A@B |\n|---|---|---|---|\n")
    for n, a, b, c in rows:
        f.write(f"| {n} | {a:.2e} | {b:.2e} | {c:.2e} |\n")
    f.write(f"\nZero-determinant demo: {d0['text']}\n")
    f.write(f"\nForward mapping holes (rotate 30, scale 1.5): {r['n_holes']} of {r['n_footprint']} pixels; inverse mapping: 0.\n")
print("figures written to", FIG)
print(open(os.path.join("tests", "validation_results.md")).read())
