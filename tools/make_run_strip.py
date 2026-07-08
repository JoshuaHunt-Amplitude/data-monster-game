"""Build the run-cycle strip from the 10-frame (5x2 grid) render.

Same pipeline as slice_sprites.py — dark-component detection, edge flood
background removal — plus a keep-largest-component pass per frame to drop
stray marks, and a 2x downscale (NEAREST) to keep the asset light.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "assets/raw/datamonster-run-v2.png"
OUT = "assets/datamonster-run.png"

img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(int)
H, W, _ = a.shape
lum = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000

dark = lum < 185
labels, n = ndimage.label(dark, structure=np.ones((3, 3)))
comps = []
for sl in ndimage.find_objects(labels):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if h >= 150 and w >= 150:
        comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
assert len(comps) >= 10, f"found {len(comps)} large components"

# group into rows, sort left-to-right
comps.sort(key=lambda b: (b[1] + b[3]) / 2)
rows = [comps[:len(comps) // 2], comps[len(comps) // 2:]]
order = []
for row in rows:
    order += sorted(row, key=lambda b: b[0])

def extract(box, pad=8):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = a[y0:y1, x0:x1]
    clum = lum[y0:y1, x0:x1]
    sat = crop.max(axis=2) - crop.min(axis=2)
    light = (clum > 200) | ((clum > 165) & (sat < 25))
    bg_labels, _ = ndimage.label(light, structure=np.ones((3, 3)))
    edge_ids = set(bg_labels[0, :]) | set(bg_labels[-1, :]) | \
               set(bg_labels[:, 0]) | set(bg_labels[:, -1])
    edge_ids.discard(0)
    alpha = np.where(np.isin(bg_labels, list(edge_ids)), 0, 255)
    # keep only the largest opaque component (drops stray marks/dust)
    op_labels, _ = ndimage.label(alpha > 0, structure=np.ones((3, 3)))
    if op_labels.max() > 1:
        sizes = ndimage.sum(alpha > 0, op_labels, range(1, op_labels.max() + 1))
        keep = int(np.argmax(sizes)) + 1
        alpha = np.where(op_labels == keep, alpha, 0)
    # reseal eye whites the edge flood ate through a thin fur gap (close the
    # narrow neck, fill the now-enclosed hole); leg/arm gaps are far wider
    m = ndimage.binary_closing(alpha > 0, iterations=3)
    m = ndimage.binary_fill_holes(m)
    alpha = np.where(m, 255, 0)
    rgba = np.dstack([crop, alpha]).astype(np.uint8)
    im = Image.fromarray(rgba)
    bbox = im.getbbox()
    return im.crop(bbox)

frames = [extract(b) for b in order]
fw = max(f.width for f in frames) + 4
fh = max(f.height for f in frames) + 4
strip = Image.new("RGBA", (fw * len(frames), fh), (0, 0, 0, 0))
for i, f in enumerate(frames):
    strip.paste(f, (i * fw + (fw - f.width) // 2, fh - f.height), f)
strip = strip.resize((strip.width // 2, strip.height // 2), Image.NEAREST)
strip.save(OUT)
print(f"{len(frames)} frames, {fw//2}x{fh//2} each -> {OUT}")
