"""Image Transformation Studio - Streamlit interface.

Run from the project root:   streamlit run app/app.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st

from batch import build_queue, process_queue
from imageio import load_image, make_sample_image, to_png_bytes, to_rgb
from session import EditorSession
from chain import STEP_NAMES
from database import Database
from matrix import SingularMatrixError, format_matrix, inverse
from transforms import about_point, compose, describe_determinant, rotation, scaling
from warp import forward_vs_inverse, forward_warp, inverse_warp

st.set_page_config(page_title="Image Transformation Studio", layout="wide")


@st.cache_resource
def get_db():
    db_path = os.environ.get("STUDIO_DB_PATH", "/tmp/studio.db")
    return Database(db_path)

db = get_db()
ss = st.session_state

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Image")
upload = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    max_upload_size=50
)
interp = st.sidebar.radio("Interpolation", ["bilinear", "nearest"],
                          help="How a value is chosen when the source point falls between pixels.")
expand = st.sidebar.checkbox("Expand canvas to fit result", value=True)

if upload is not None:
    key = f"{upload.name}-{upload.size}"
    if ss.get("loaded_key") != key:
        ss.editor = EditorSession(db, load_image(upload.getvalue()), upload.name)
        ss.loaded_key = key
if st.sidebar.button("Use sample image"):
    ss.editor = EditorSession(db, make_sample_image(), "sample.png")
    ss.loaded_key = None

st.sidebar.header("2. Reopen a saved session")
saved = db.list_images()
if saved:
    labels = {f"#{r['id']}  {r['name']}  ({r['steps']} edits)": r["id"] for r in saved}
    choice = st.sidebar.selectbox("Saved sessions", list(labels))
    if st.sidebar.button("Open session"):
        ss.editor = EditorSession.from_db(db, labels[choice])
        ss.loaded_key = None
else:
    st.sidebar.caption("No saved sessions yet.")

ed = ss.get("editor")
if ed is not None:
    ed.interpolation, ed.expand = interp, expand
demo_img = ed.original if ed is not None else make_sample_image()

st.title("Image Transformation Studio")
st.caption("Every edit is a matrix. Output pixels are found with the INVERSE matrix, so there are no holes.")
tab_studio, tab_inv, tab_zero, tab_batch = st.tabs(
    ["Studio", "Why inverse mapping?", "Zero determinant", "Batch (queue)"])

# ------------------------------------------------------------------ studio
with tab_studio:
    if ed is None:
        st.info("Upload an image or click 'Use sample image' in the sidebar to begin.")
    else:
        kind = st.selectbox("Edit", STEP_NAMES)
        params = {}
        if kind == "Rotate":
            params["angle"] = st.slider("Angle (degrees, counter-clockwise)", -180.0, 180.0, 30.0, 1.0)
        elif kind == "Shear":
            params["kx"] = st.slider("Horizontal shear kx  (x' = x + kx*y)", -1.5, 1.5, 0.3, 0.05)
            params["ky"] = st.slider("Vertical shear ky  (y' = y + ky*x)", -1.5, 1.5, 0.0, 0.05)
        elif kind == "Scale":
            c1, c2 = st.columns(2)
            params["sx"] = c1.number_input("Scale x", value=1.5, step=0.1, format="%.2f")
            params["sy"] = c2.number_input("Scale y", value=1.5, step=0.1, format="%.2f")

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Apply edit", type="primary"):
            try:
                ed.apply(kind, params)
            except (SingularMatrixError, ValueError) as e:
                st.error(str(e))
        if b2.button("Undo", disabled=not ed.can_undo()):
            ed.undo()
            st.rerun()
        if b3.button("Redo", disabled=not ed.can_redo()):
            ed.redo()
            st.rerun()
        if b4.button("Save result to database"):
            st.success(f"Saved result #{ed.save_result()}")

        out = ed.render()
        left, right = st.columns(2)
        left.image(ed.original, caption=f"Before ({ed.width} x {ed.height})")
        right.image(out, caption=f"After ({out.shape[1]} x {out.shape[0]}), {interp} interpolation")
        st.download_button("Download result (PNG)", to_png_bytes(out), "result.png", "image/png")

        st.subheader("History and combined matrix")
        steps = ed.active_steps()
        if steps:
            st.dataframe([{"Step": s.step_no, "Edit": s.name,
                           "Parameters": ", ".join(f"{k}={v}" for k, v in s.params.items()) or "-",
                           "det": round(s.det, 4)} for s in steps], hide_index=True)
            total = ed.total_matrix()
            st.code("Combined matrix (last edit x ... x first edit):\n" + format_matrix(total))
            st.info(describe_determinant(total)["text"])
        else:
            st.caption("No edits yet. The combined matrix is the identity.")
        st.caption(f"Redo stack holds {len(ed.redo_stack)} undone edit(s).")

# ------------------------------------------------------------ inverse mapping
with tab_inv:
    st.subheader("Forward mapping leaves holes; inverse mapping does not")
    a = st.slider("Rotation angle", -180, 180, 30, key="inv_a")
    s = st.slider("Uniform scale", 0.5, 3.0, 1.0, 0.1, key="inv_s")
    h, w = demo_img.shape[:2]
    M = about_point(compose([rotation(a), scaling(s, s)]), (w - 1) / 2, (h - 1) / 2)
    try:
        r = forward_vs_inverse(demo_img, M, expand=True)
        marked = to_rgb(r["forward"]).copy()
        marked[r["holes"]] = (255, 0, 0)
        c1, c2, c3 = st.columns(3)
        c1.image(r["forward"], caption="Forward mapping (push input pixels)")
        c2.image(marked, caption="Holes shown in red")
        c3.image(r["inverse"], caption="Inverse mapping (pull from input)")
        st.metric("Holes left by forward mapping", f"{r['n_holes']:,}",
                  f"{100 * r['n_holes'] / max(r['n_footprint'], 1):.1f}% of the result")
        st.write("Inverse mapping starts from each OUTPUT pixel, so every one gets a value: **0 holes**.")
        st.code("M^-1 =\n" + format_matrix(inverse(M)))
    except ValueError as e:
        st.error(str(e))

# ------------------------------------------------------------ zero determinant
with tab_zero:
    st.subheader("What happens when det = 0?")
    preset = st.radio("Matrix", ["Squash y to zero (scale y by 0)", "Custom 2x2 matrix"], horizontal=True)
    if preset.startswith("Squash"):
        m2 = [[1.0, 0.0], [0.0, 0.0]]
    else:
        cc = st.columns(4)
        vals = [cc[i].number_input(n, value=v, step=0.5) for i, (n, v) in
                enumerate(zip("abcd", [1.0, 2.0, 0.5, 1.0]))]
        m2 = [[vals[0], vals[1]], [vals[2], vals[3]]]
    h, w = demo_img.shape[:2]
    M = about_point([[m2[0][0], m2[0][1], 0.0], [m2[1][0], m2[1][1], 0.0], [0.0, 0.0, 1.0]],
                    (w - 1) / 2, (h - 1) / 2)
    info = describe_determinant(M)
    st.code("2x2 part =\n" + format_matrix(m2))
    st.write(info["text"])
    try:
        st.image(inverse_warp(demo_img, M, "bilinear", expand=False), caption="Inverse mapping result")
    except SingularMatrixError as e:
        st.error(f"Inverse mapping is impossible: {e}")
        fwd, _ = forward_warp(demo_img, M, expand=False)
        st.image(fwd, caption="Forward mapping shows what happens: every pixel lands on a line, "
                              "so the picture cannot be recovered.")
    st.caption("Different input pixels now land on the same output pixel, so no inverse matrix can "
               "separate them again. This is why the app rejects an edit with det = 0.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Apply the current edit chain to many images")
    files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg", "bmp", "webp"],
                             accept_multiple_files=True, key="batch_files ,  max_upload_size=50")
    
    chain = ed.chain() if ed is not None else []
    st.write(f"Current chain: {', '.join(n for n, _ in chain) or '(empty - identity)'}")
    if files and st.button("Process queue"):
        jobs = build_queue([(f.name, load_image(f.getvalue())) for f in files])
        st.write(f"{len(jobs)} job(s) queued; processing first-in, first-out.")
        for name, result in process_queue(jobs, chain, interp, expand):
            st.image(result, caption=f"{name} -> {result.shape[1]} x {result.shape[0]}")
            st.download_button(f"Download {name}", to_png_bytes(result), f"edited_{name}.png",
                               "image/png", key=f"dl_{name}")
