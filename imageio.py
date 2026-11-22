"""Image file input/output. Pillow is used ONLY to read, write and downscale files."""
import io

import numpy as np
from PIL import Image

MAX_SIDE_UPLOAD = 1024


def load_image(data, max_side=MAX_SIDE_UPLOAD):
    """bytes -> uint8 array (H x W for grayscale, H x W x 3 for colour)."""
    im = Image.open(io.BytesIO(data))
    im = im.convert("L") if im.mode in ("L", "1") else im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.array(im, dtype=np.uint8)


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def from_png_bytes(data):
    return np.array(Image.open(io.BytesIO(data)), dtype=np.uint8)


def to_rgb(arr):
    return np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr


def make_sample_image(w=320, h=240):
    """Asymmetric test picture, so flips and rotations are easy to see."""
    from PIL import ImageDraw
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 // w, y * 255 // h, np.full_like(x, 150)], -1).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for gx in range(0, w, 40):
        d.line([(gx, 0), (gx, h)], fill=(255, 255, 255), width=1)
    for gy in range(0, h, 40):
        d.line([(0, gy), (w, gy)], fill=(255, 255, 255), width=1)
    d.rectangle([40, 40, 60, 160], fill=(255, 220, 0))       # letter "F"
    d.rectangle([40, 40, 130, 60], fill=(255, 220, 0))
    d.rectangle([40, 90, 100, 108], fill=(255, 220, 0))
    d.ellipse([200, 120, 280, 200], fill=(220, 30, 60))      # circle
    d.polygon([(230, 30), (290, 30), (230, 90)], fill=(20, 20, 20))  # corner triangle
    return np.array(im, dtype=np.uint8)
